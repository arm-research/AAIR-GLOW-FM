# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist


def _chw_to_uint8_hwc(x: np.ndarray) -> np.ndarray:
    """
    x: (C,H,W) numpy, float in [0,1] or [-1,1] or uint8
    returns: (H,W,3) uint8
    """
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0 + 0.5).astype(np.uint8)

    # CHW -> HWC
    x = np.transpose(x, (1, 2, 0))

    # force 3 channels
    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    elif x.shape[2] > 3:
        x = x[:, :, :3]
    return x


def _concat_h(images, pad=6):
    H = images[0].shape[0]
    spacer = np.zeros((H, pad, 3), dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.concatenate([out, spacer, im], axis=1)
    return out


def _concat_v(images, pad=10):
    W = images[0].shape[1]
    spacer = np.zeros((pad, W, 3), dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.concatenate([out, spacer, im], axis=0)
    return out


def log_sr_panels_from_res_dict(
    *,
    exp,
    run_id: str,
    res_imgs_by_step: dict,
    artifact_path: str,
    epoch: int,
    steps=(1, 2, 4),
    max_k: int = 8,
    saved_res: int | None = None,
):
    """
    Logs up to `max_k` panels inferred from batch size.
    res_imgs_by_step: {step: {"img_lq":(B,C,H,W), "img_hq":..., "img_hq_estimate":...}}

    saved_res:
        If not None, resize the *saved* panel so that each image tile's height becomes
        (saved_res * 2). Example: saved_res=256 -> tiles saved at 512px height.
        (Uses bicubic resizing.)
    """
    for s in steps:
        if s not in res_imgs_by_step:
            raise ValueError(f"Missing step={s} in res_imgs_by_step")
        d = res_imgs_by_step[s]
        for key in ("img_lq", "img_hq", "img_hq_estimate"):
            if key not in d:
                raise ValueError(f"Missing key '{key}' for step={s}")

    B = res_imgs_by_step[steps[0]]["img_lq"].shape[0]
    k = min(max_k, B)

    # Determine resize scale (based on the underlying image tile resolution)
    scale = None
    if saved_res is not None:
        if saved_res <= 0:
            raise ValueError("saved_res must be a positive int or None")
        # Use the lq tensor's H as the reference "tile" height
        ref_h = int(res_imgs_by_step[steps[0]]["img_lq"].shape[-2])
        target_tile_h = int(saved_res) * 2
        if ref_h > 0 and target_tile_h > 0:
            scale = target_tile_h / ref_h

    with tempfile.TemporaryDirectory() as td:
        for i in range(k):
            rows = []
            for s in steps:
                d = res_imgs_by_step[s]
                lq = _chw_to_uint8_hwc(d["img_lq"][i])
                sr = _chw_to_uint8_hwc(d["img_hq_estimate"][i])
                gt = _chw_to_uint8_hwc(d["img_hq"][i])
                rows.append(_concat_h([lq, sr, gt], pad=6))

            panel = _concat_v(rows, pad=10)
            img = Image.fromarray(panel)

            # Resize only for saving (doesn't affect what gets logged as "content")
            if scale is not None and scale != 1.0:
                new_w = max(1, int(round(img.width * scale)))
                new_h = max(1, int(round(img.height * scale)))
                img = img.resize((new_w, new_h), resample=Image.BICUBIC)

            fname = f"val_{i:03d}_epoch_{epoch:04d}.png"
            fpath = os.path.join(td, fname)
            img.save(fpath, format="PNG")

            try:
                exp.log_artifact(
                    run_id,
                    fpath,
                    artifact_path=f"{artifact_path}/epoch_{epoch:04d}",
                )
            except Exception as e:
                print(f"[rank] log_sr_panels_from_res_dict: log_artifact failed: {e}", flush=True)


def dump_tensor(x: torch.Tensor, name: str, step: int, rank: int, out_dir="/tmp/bad_tensors"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_rank{rank}_step{step}.pt")
    # move small-ish copy to cpu to avoid GPU issues
    torch.save({
        "name": name,
        "step": step,
        "rank": rank,
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "tensor": x.detach().float().cpu(),
    }, path)
    print(f"[rank{rank}] saved {name} -> {path}")


def rank_print(msg: str):
    r = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[rank {r}] {msg}", flush=True)

def check_metric_device(metric_module, expected_device: torch.device, name="lpips"):
    # expected_device should be trainer.strategy.root_device or self.device
    if metric_module is None:
        rank_print(f"{name}: metric_module is None")
        return

    # find one parameter as a representative
    try:
        p = next(metric_module.parameters())
        actual = p.device
        dtype = p.dtype
    except StopIteration:
        # some wrappers may have no parameters at top-level, fallback to buffers
        b = next(metric_module.buffers())
        actual = b.device
        dtype = b.dtype

    rank_print(
        f"{name}: expected={expected_device}, actual={actual}, dtype={dtype}, "
        f"cuda.current_device()={torch.cuda.current_device()}, "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
    )
