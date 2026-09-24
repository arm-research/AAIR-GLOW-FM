# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import argparse
import copy
import math
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API\\..*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="Found .* module\\(s\\) in eval mode at the start of training\\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"It is recommended to use `self\.log\('val_.*sync_dist=True\)` when logging on epoch level in distributed setting to accumulate the metric across devices\.",
    category=UserWarning,
)
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from datasets.DLL import (
    BucketAwareSubset,
    DDRTrainDataset,
    DDRValDataset,
    PairedImageTrainDataset,
    PairedImageValDataset,
    RandomBucketBatchSampler,
    random_bucket_policy_from_dataset,
)
from models.generate_images import generator_fn, generator_fn_fm_only
from models.losses import DistortionLoss, EndpointRGBLoss, RGBReconstructionLosses
from models.metrics import GT_MEAN_MATCH_METRICS, MetricEval, match_prediction_mean_to_gt
from utils import resolver


class BalancedMultiTrainDataset(Dataset):
    """Round-robin wrapper that samples each child dataset equally per epoch."""

    def __init__(self, datasets: List[Dataset], names: Optional[List[str]] = None):
        if not datasets:
            raise ValueError("BalancedMultiTrainDataset requires at least one dataset")
        self.datasets = list(datasets)
        self.names = names or [f"dataset_{idx}" for idx in range(len(datasets))]
        self.lengths = [len(dataset) for dataset in self.datasets]
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"All balanced train datasets must be non-empty, got {dict(zip(self.names, self.lengths))}")
        self.max_len = max(self.lengths)
        self.total_len = self.max_len * len(self.datasets)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        bucket_shape = None
        if isinstance(index, tuple) and len(index) == 3:
            index, crop_h, crop_w = index
            bucket_shape = (int(crop_h), int(crop_w))
        dataset_idx = index % len(self.datasets)
        sample_idx = (index // len(self.datasets)) % self.lengths[dataset_idx]
        if bucket_shape is not None:
            return self.datasets[dataset_idx][(sample_idx, bucket_shape[0], bucket_shape[1])]
        return self.datasets[dataset_idx][sample_idx]


def torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_model_init_checkpoint(net, init_cfg):
    ckpt_path = init_cfg.get("ckpt_path", None)
    if not ckpt_path:
        return

    ckpt_path = os.fspath(ckpt_path)
    if ckpt_path.startswith("s3://"):
        raise ValueError(
            "Remote S3 init checkpoints are not supported in this submission path. "
            "Please provide a local init.ckpt_path or remove init.ckpt_path."
        )
    checkpoint = torch_load_checkpoint(ckpt_path)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value

    if not model_state:
        model_state = state_dict

    target_state = net.state_dict()
    ignore_mismatched_shapes = bool(init_cfg.get("ignore_mismatched_shapes", True))
    if ignore_mismatched_shapes:
        model_state = {
            key: value
            for key, value in model_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }

    missing, unexpected = net.load_state_dict(model_state, strict=bool(init_cfg.get("strict", False)))
    print(
        f"Initialized JVP model from {ckpt_path}: loaded={len(model_state)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

def compute_rgb_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction.float() - target.float()).square(), dim=(1, 2, 3))
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-12))

def _plain_cfg(obj):
    if obj is None:
        return None
    if isinstance(obj, (dict, list, tuple, str, int, float, bool)):
        return obj
    try:
        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return obj

def _cfg_list(entry, key: str) -> List[str]:
    value = entry.get(key, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)

def _resolve_dataset_dir(entry, path: str, single_key: str, candidates_key: str, default: str) -> str:
    candidates = []
    if entry.get(single_key, None) is not None:
        candidates.append(entry[single_key])
    candidates.extend(_cfg_list(entry, candidates_key))
    if not candidates:
        candidates.extend(default if isinstance(default, (list, tuple)) else [default])

    for candidate in candidates:
        full_path = candidate if os.path.isabs(candidate) else os.path.join(path, candidate)
        if os.path.isdir(full_path):
            return candidate
    return candidates[0]

def _paired_default_dirs(kind: str, is_train: bool):
    if kind in {"efficient_llie", "efficient_llie_ds", "efficient-llie", "efficient-llie-ds"}:
        if is_train:
            return (
                ["Train/Input", "train/input", "Train/Low", "train/low", "Input", "Low", "input", "low"],
                ["Train/GT", "train/gt", "Train/Normal", "train/normal", "GT", "Normal", "gt", "high"],
            )
        return (
            ["Test/Input", "test/input", "Test/Low", "test/low", "Input", "Low", "input", "low"],
            ["Test/GT", "test/gt", "Test/Normal", "test/normal", "GT", "Normal", "gt", "high"],
        )
    if kind in {"nll", "lsd_nll", "lsd-nll"}:
        if is_train:
            return (
                ["Train/Input", "Train/Low", "train/input", "train/low", "inputPatchNLL", "Input", "Low", "input", "low"],
                ["Train/GT", "Train/Normal", "train/gt", "train/normal", "gtPatchNLL", "GT", "Normal", "gt", "high"],
            )
        return (
            ["Val/Input", "Val/Low", "val/input", "val/low", "inputPatchNLL", "Input", "Low", "input", "low"],
            ["Val/GT", "Val/Normal", "val/gt", "val/normal", "gtPatchNLL", "GT", "Normal", "gt", "high"],
        )
    if kind in {"uhd_lol4k", "uhd_lol_4k", "uhd-lol4k", "uhd_lol"}:
        if is_train:
            return (
                ["Train/low", "Train/Input", "Train/Low", "train/low", "train/input", "low", "input", "LQ", "Low"],
                ["Train/high", "Train/GT", "Train/Normal", "train/high", "train/gt", "train/normal", "high", "gt", "Normal", "GT"],
            )
        return (
            ["Test/low", "Test/Input", "Test/Low", "test/low", "test/input", "low", "input", "LQ", "Low"],
            ["Test/high", "Test/GT", "Test/Normal", "test/high", "test/gt", "test/normal", "high", "gt", "Normal", "GT"],
        )
    if kind in {"sdsd", "sdsd_indoor", "sdsd_outdoor", "sdsd_static"}:
        if is_train:
            return (
                ["Train/Low", "train/low", "Train/Input", "train/input", "input", "LQ", "Low", "low"],
                ["Train/Normal", "train/normal", "Train/GT", "train/gt", "GT", "Normal", "high", "gt"],
            )
        return (
            ["Test/Low", "test/low", "Test/Input", "test/input", "input", "LQ", "Low", "low"],
            ["Test/Normal", "test/normal", "Test/GT", "test/gt", "GT", "Normal", "high", "gt"],
        )
    if kind in {"lsrw", "lswr"}:
        if is_train:
            return (
                ["Train/Low", "train/low", "Train/Input", "train/input", "Low", "low"],
                ["Train/Normal", "train/normal", "Train/GT", "train/gt", "Normal", "high", "gt"],
            )
        return (
            ["Test/Low", "test/low", "Test/Input", "test/input", "Low", "low"],
            ["Test/Normal", "test/normal", "Test/GT", "test/gt", "Normal", "high", "gt"],
        )
    return ["low"], ["gt"]

def _train_crop_padding_kwargs(entry, defaults=None):
    defaults = defaults or {}
    return {
        "pad_to_patch_size": bool(
            entry.get(
                "pad_to_patch_size",
                entry.get("pad_if_needed", defaults.get("pad_to_patch_size", defaults.get("pad_if_needed", False))),
            )
        ),
        "crop_padding_mode": str(entry.get("crop_padding_mode", defaults.get("crop_padding_mode", "reflect"))),
    }

def _train_crop_size_kwargs(entry, defaults=None):
    defaults = defaults or {}
    out = {}
    for key in ("crop_h", "crop_w"):
        value = entry.get(key, defaults.get(key, None))
        if value is not None:
            out[key] = int(value)
    return out

def _train_crop_dims_configured(entry, defaults=None) -> bool:
    return bool(_train_crop_size_kwargs(_plain_cfg(entry) or {}, _plain_cfg(defaults) or {}))

def _merge_nested_dicts(base, override):
    out = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_nested_dicts(out[key], value)
        else:
            out[key] = value
    return out

def _train_augment_kwargs(entry, defaults=None):
    entry = _plain_cfg(entry) or {}
    defaults = _plain_cfg(defaults) or {}
    default_aug = defaults.get("augment", None)
    entry_aug = entry.get("augment", None)
    if default_aug is None and entry_aug is None:
        return {}

    augment = _merge_nested_dicts(default_aug or {}, entry_aug or {})
    if entry_aug is None and "use_aug" in entry:
        augment["enabled"] = bool(entry.get("use_aug"))
    return {"augment": augment}

def _train_random_bucket_crop_kwargs(entry, defaults=None):
    entry = _plain_cfg(entry) or {}
    defaults = _plain_cfg(defaults) or {}
    default_bucket = defaults.get("random_bucket_crop", None)
    entry_bucket = entry.get("random_bucket_crop", None)
    if default_bucket is None and entry_bucket is None:
        return {}
    return {"random_bucket_crop": _merge_nested_dicts(default_bucket or {}, entry_bucket or {})}

def _maybe_bucket_loader_kwargs(dataset, batch_size, shuffle, drop_last, cfg=None):
    policy = random_bucket_policy_from_dataset(dataset)
    if policy is None:
        return {"batch_size": batch_size, "shuffle": shuffle, "drop_last": drop_last}
    rank, world_size = _rank_world_for_loader(cfg) if cfg is not None else _distributed_rank_world()
    if world_size > 1:
        base_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return {
        "batch_sampler": RandomBucketBatchSampler(
            base_sampler,
            batch_size=batch_size,
            policy=policy,
            drop_last=drop_last,
        )
    }

def _random_bucket_crop_enabled_cfg(cfg) -> bool:
    dataset_cfg = _plain_cfg(getattr(cfg, "dataset", {})) or {}
    configs = [dataset_cfg]
    configs.extend(_plain_cfg(entry) or {} for entry in dataset_cfg.get("train_sets", []) or [])
    for entry in configs:
        bucket_cfg = _plain_cfg(entry.get("random_bucket_crop", None)) or {}
        augment_cfg = _plain_cfg(entry.get("augment", None)) or {}
        augment_bucket = _plain_cfg(augment_cfg.get("random_bucket_crop", None)) or {}
        if bool(bucket_cfg.get("enabled", False) or augment_bucket.get("enabled", False)):
            return True
    return False


def _sync_dist_logs_enabled(cfg) -> bool:
    training_cfg = getattr(cfg, "training", OmegaConf.create({}))
    configured = getattr(training_cfg, "sync_dist_logs", None)
    if configured is not None:
        return bool(configured)
    return not _random_bucket_crop_enabled_cfg(cfg)


def _planned_num_devices(cfg) -> int:
    devices = getattr(getattr(cfg, "pl", OmegaConf.create({})), "devices", 1)
    if isinstance(devices, int):
        return max(1, int(devices))
    if isinstance(devices, str):
        value = devices.strip().lower()
        return 1 if value in {"", "auto"} else len([d for d in devices.split(",") if d.strip()])
    try:
        return max(1, len(devices))
    except TypeError:
        return 1


def _rank_world_for_loader(cfg) -> tuple[int, int]:
    rank, world_size = _distributed_rank_world()
    if world_size > 1:
        return rank, world_size
    planned_world = _planned_num_devices(cfg)
    if planned_world <= 1:
        return rank, world_size
    planned_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return planned_rank, planned_world


def _disable_encoder_compile_for_random_buckets(cfg) -> None:
    training_cfg = getattr(cfg, "training", OmegaConf.create({}))
    if bool(getattr(training_cfg, "allow_random_bucket_encoder_compile", False)):
        print(
            "[random_bucket_crop] compile paths remain enabled because "
            "training.allow_random_bucket_encoder_compile=true",
            flush=True,
        )
        return
    encoder_cfg = getattr(cfg, "encoder", None)
    if encoder_cfg is not None and bool(getattr(encoder_cfg, "use_compile", False)):
        encoder_cfg.use_compile = False
    os.environ["K_DIFFUSION_USE_COMPILE"] = "0"
    print(
        "[random_bucket_crop] disabled encoder.use_compile and K_DIFFUSION_USE_COMPILE for "
        "random bucket training; cold torch.compile/autotune with many crop shapes can stall DDP. "
        "Set training.allow_random_bucket_encoder_compile=true to opt in.",
        flush=True,
    )


def _distributed_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size

def build_train_dataset(entry, patch_size, return_uint, defaults=None):
    entry = _plain_cfg(entry) or {}
    defaults = _plain_cfg(defaults) or {}
    kind = str(entry.get("type", "paired")).lower()
    path = entry.get("path", entry.get("root", ""))
    use_aug = bool(entry.get("use_aug", True))
    augment_kwargs = _train_augment_kwargs(entry, defaults)

    if kind in {"dll", "ddr"}:
        return DDRTrainDataset(
            data_root=path,
            patch_size=patch_size,
            use_aug=use_aug,
            return_uint8=return_uint,
            preprocess_file_path=entry.get("preprocess_file_path", ""),
            **_train_crop_size_kwargs(entry, defaults),
            **_train_crop_padding_kwargs(entry, defaults),
            **augment_kwargs,
            **_train_random_bucket_crop_kwargs(entry, defaults),
        )

    if kind in {
        "paired", "folder", "lol", "lolv1", "lolv2", "lolv2_real", "lolv2_sim",
        "lolv2_syn", "lolv2_synthetic", "lsrw", "lswr",
        "sdsd", "sdsd_indoor", "sdsd_outdoor", "sdsd_static",
        "uhd_lol4k", "uhd_lol_4k", "uhd-lol4k", "uhd_lol",
        "nll", "lsd_nll", "lsd-nll",
        "efficient_llie", "efficient_llie_ds", "efficient-llie", "efficient-llie-ds",
    }:
        default_low_dirs, default_gt_dirs = _paired_default_dirs(kind, is_train=True)
        return PairedImageTrainDataset(
            data_root=path,
            patch_size=patch_size,
            low_dir=_resolve_dataset_dir(entry, path, "low_dir", "low_dirs", default_low_dirs),
            gt_dir=_resolve_dataset_dir(entry, path, "gt_dir", "gt_dirs", entry.get("high_dir", default_gt_dirs)),
            return_uint8=return_uint,
            use_aug=use_aug,
            strict_pairs=bool(entry.get("strict_pairs", True)),
            **_train_crop_size_kwargs(entry, defaults),
            **_train_crop_padding_kwargs(entry, defaults),
            **augment_kwargs,
            **_train_random_bucket_crop_kwargs(entry, defaults),
        )

    raise ValueError(f"Unsupported train dataset type: {kind}")

def build_val_dataset(entry, default_path, patch_size, return_uint):
    entry = _plain_cfg(entry) or {}
    kind = str(entry.get("type", "dll")).lower()
    path = entry.get("path", entry.get("root", default_path))

    if kind in {"dll", "ddr"}:
        return DDRValDataset(
            path,
            patch_size=patch_size,
            return_uint8=return_uint,
            preprocess_file_path=entry.get("preprocess_file_path", ""),
        )

    if kind in {
        "paired", "folder", "lol", "lolv1", "lolv2", "lolv2_real", "lolv2_sim",
        "lolv2_syn", "lolv2_synthetic", "lsrw", "lswr",
        "sdsd", "sdsd_indoor", "sdsd_outdoor", "sdsd_static",
        "uhd_lol4k", "uhd_lol_4k", "uhd-lol4k", "uhd_lol",
        "nll", "lsd_nll", "lsd-nll",
        "efficient_llie", "efficient_llie_ds", "efficient-llie", "efficient-llie-ds",
    }:
        default_low_dirs, default_gt_dirs = _paired_default_dirs(kind, is_train=False)
        return PairedImageValDataset(
            data_root=path,
            patch_size=patch_size,
            low_dir=_resolve_dataset_dir(entry, path, "low_dir", "low_dirs", default_low_dirs),
            gt_dir=_resolve_dataset_dir(entry, path, "gt_dir", "gt_dirs", entry.get("high_dir", default_gt_dirs)),
            return_uint8=return_uint,
            strict_pairs=bool(entry.get("strict_pairs", True)),
        )

    raise ValueError(f"Unsupported val dataset type: {kind}")

def needs_padding(H, W, mult):
    return (H % mult) != 0 or (W % mult) != 0

def pad_latent_to_multiple(x, mult=32, mode="reflect"):
    # x: [B,C,H,W]
    B, C, H, W = x.shape
    pad_h = (mult - H % mult) % mult
    pad_w = (mult - W % mult) % mult
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)  # (left,right,top,bottom)
    meta = (H, W, pad_h, pad_w)
    return x, meta


def unpad_latent(x, meta):
    H, W, pad_h, pad_w = meta
    return x[..., :H, :W]


def unpad_image(x, meta):
    if meta is None:
        return x
    H0, W0 = meta["orig_hw"]
    return x[..., :H0, :W0]

def pad_image_if_needed(x, mult=8, mode="reflect"):
    """
    Pads image tensor [B,C,H,W] to multiple of `mult`.
    Returns:
        x_padded
        meta or None
    """
    B, C, H, W = x.shape

    pad_h = (mult - H % mult) % mult
    pad_w = (mult - W % mult) % mult

    if pad_h == 0 and pad_w == 0:
        return x, None

    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    meta = {
        "orig_hw": (H, W),
    }
    return x, meta



def is_rank0_process():
    # Works under torchrun and Lightning DDP spawn
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    return rank == 0

def _cuda_device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    if device.index is not None:
        return int(device.index)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)
    return torch.cuda.current_device() if torch.cuda.is_available() else None

def _dist_barrier(device: Optional[torch.device] = None):
    if not (dist.is_available() and dist.is_initialized()):
        return
    device_idx = _cuda_device_index(device) if device is not None else None
    if device_idx is not None:
        try:
            dist.barrier(device_ids=[device_idx])
            return
        except TypeError:
            pass
    dist.barrier()

def _ddp_debug_enabled(cfg) -> bool:
    env_value = os.environ.get("TVM_DDP_DEBUG", "")
    if env_value:
        return env_value.strip().lower() not in {"0", "false", "no", "off"}
    debug_cfg = getattr(cfg, "debug", OmegaConf.create({}))
    return bool(getattr(debug_cfg, "ddp_device_log", False))

def _sync_cuda_for_debug(device: torch.device):
    if torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)

def build_train_loader(cfg, batch_gpu, return_uint):
    debug_cfg = cfg.get("debug", {})
    overfit_test = bool(debug_cfg.get("overfit_test", False))
    train_sets = cfg.dataset.get("train_sets", None)
    if train_sets is None:
        train_dset = DDRTrainDataset(
            data_root=cfg.dataset.path,
            patch_size=cfg.ir.input_res,
            use_aug=True,
            return_uint8=return_uint,
            **_train_crop_size_kwargs(cfg.dataset, cfg.dataset),
            pad_to_patch_size=bool(cfg.dataset.get("pad_to_patch_size", cfg.dataset.get("pad_if_needed", False))),
            crop_padding_mode=str(cfg.dataset.get("crop_padding_mode", "reflect")),
            **_train_augment_kwargs(cfg.dataset, cfg.dataset),
            **_train_random_bucket_crop_kwargs(cfg.dataset, cfg.dataset),
        )
    else:
        train_sets = [_plain_cfg(entry) for entry in train_sets]
        train_dsets = [
            build_train_dataset(entry, patch_size=cfg.ir.input_res, return_uint=return_uint, defaults=cfg.dataset)
            for entry in train_sets
        ]
        if len(train_dsets) == 1:
            train_dset = train_dsets[0]
        else:
            names = [str(entry.get("name", f"dataset_{idx}")) for idx, entry in enumerate(train_sets)]
            # BalancedMultiTrainDataset equalizes sampling across sources, so epoch length is no longer
            # just the sum of raw dataset sizes. Printing the derived lengths helps explain "fast" epochs.
            train_dset = BalancedMultiTrainDataset(train_dsets, names=names)
            print(
                "[dataset] balanced multi-train: "
                + ", ".join(f"{name}={length}" for name, length in zip(names, train_dset.lengths))
                + f"; epoch_len={len(train_dset)}",
                flush=True,
            )

    train_subset_size = int(cfg.dataset.get("train_subset_size", 0) or 0)
    if train_subset_size > 0:
        subset_k = min(train_subset_size, len(train_dset))
        train_dset = BucketAwareSubset(train_dset, range(subset_k))
        print(f"[train] train_subset_size={train_subset_size}: using first {subset_k} training samples", flush=True)

    if overfit_test:
        overfit_k = min(int(debug_cfg.get("overfit_k", 1)), len(train_dset))
        if overfit_k <= 0:
            raise ValueError(f"debug.overfit_k must be positive, got {overfit_k}")
        train_dset = BucketAwareSubset(train_dset, range(overfit_k))
        print(f"[debug] overfit_test=true: using first {overfit_k} training samples", flush=True)

    return DataLoader(
        train_dset,
        **_maybe_bucket_loader_kwargs(
            train_dset,
            batch_gpu,
            shuffle=True if not overfit_test else False,
            drop_last=True if not overfit_test else False,
            cfg=cfg,
        ),
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        persistent_workers=cfg.dataloader.num_workers > 0,
    )


def _periodic_epoch_due(current_epoch: int, every_n_epochs: int) -> bool:
    every_n_epochs = int(every_n_epochs)
    if every_n_epochs <= 0:
        return False
    return int(current_epoch) > 0 and int(current_epoch) % every_n_epochs == 0


def _normalize_validation_sampler(mode: Any) -> str:
    mode = str(mode or "fm_only").lower().replace("-", "_")
    aliases = {
        "fm": "fm_only",
        "diag": "fm_only",
        "diagonal": "fm_only",
        "fm_diag": "fm_only",
        "fm_only": "fm_only",
        "jump": "jump",
        "tvm": "jump",
        "offdiag": "jump",
        "off_diagonal": "jump",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported validation sampler {mode!r}; expected 'fm_only' or 'jump'.")
    return aliases[mode]


def _validation_sampler_list(validation_cfg: Any, primary_sampler: str) -> List[str]:
    samplers = getattr(validation_cfg, "samplers", None)
    if samplers is None:
        return [primary_sampler]

    out: List[str] = []
    for sampler in samplers:
        normalized = _normalize_validation_sampler(sampler)
        if normalized not in out:
            out.append(normalized)
    if primary_sampler not in out:
        out.append(primary_sampler)
    return out

def _validation_metric_name(
    metric_name: str,
    sampler_name: str,
    *,
    num_steps: Optional[int] = None,
    include_num_steps: bool = False,
    val_name: Optional[str] = None,
) -> str:
    # Keep validation metric names stable across single-loader and multi-loader runs so the stdout
    # summaries and any downstream log parsing do not depend on Lightning's internal naming.
    parts = ["val"]
    if val_name:
        parts.append(str(val_name).lower().replace("-", "_"))
    parts.append(str(metric_name).lower().replace("/", "_"))
    if include_num_steps and num_steps is not None:
        parts.extend(["num_steps", str(int(num_steps))])
    parts.append(f"{sampler_name}_sampler")
    return "_".join(parts)



@torch.no_grad()
def _broadcast_module(module: torch.nn.Module, src: int = 0):
    # Broadcast parameters
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    # Broadcast buffers too (e.g., pos_embed if it were a buffer; BN running stats, etc.)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def setup_cudnn_and_tf32(cfg):

    cudnn_benchmark = cfg.training.cudnn_benchmark
    enable_tf32 = cfg.training.enable_tf32
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    if not cudnn_benchmark:
        torch.use_deterministic_algorithms(True)

    # Enable these to speed up on A100 GPUs
    print("enable_tf32", enable_tf32)
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,  # final_lr = base_lr * min_lr_ratio
):
    """
    Warmup + cosine decay scheduler (per optimizer step).
      - warmup: lr ramps 0 -> base_lr over warmup_steps
      - cosine: lr decays base_lr -> base_lr*min_lr_ratio over remaining steps
    """
    warmup_steps = int(max(0, warmup_steps))
    total_steps = int(max(1, total_steps))
    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        # step starts at 0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps == warmup_steps:
            return float(min_lr_ratio)

        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

class TrainJVPLightningModule(pl.LightningModule):
    """
    Wraps the TVM model & loss into a LightningModule.
    """

    def __init__(self, cfg):
        super().__init__()
        # Save config as hyperparameters (nice for checkpoints & logging)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        # Encoder
        excluded = {"res", "dim_latent"}
        encoder_kwargs = {k: v for k, v in cfg.encoder.items() if k not in excluded}
        self.encoder = resolver.instantiate(**encoder_kwargs)

        # Model
        interface_kwargs = dict(
            img_resolution=cfg.encoder.res,  # dataset_obj.resolution,
            img_channels=cfg.encoder.dim_latent,
        )
        skip_init = not is_rank0_process()  # do init only on rank 0 to save time/memory
        rank = int(os.environ.get("RANK", "0"))
        print(f"[rank {rank}] DiT skip_init={skip_init}", flush=True)
        t0 = time.time()
        print(f"[rank {rank}] building net...", flush=True)
        net = resolver.instantiate(**cfg.network, **interface_kwargs, skip_init=skip_init)
        print(f"[rank {rank}] net built in {time.time() - t0:.1f}s", flush=True)

        init_cfg = cfg.get("init", None)
        if init_cfg is not None and is_rank0_process():
            load_model_init_checkpoint(net, init_cfg)

        if cfg.network.get('compile', False):
            torch._dynamo.config.optimize_ddp = False
            net = torch.compile(net, mode="default")

        self.model = net
        self._log_to_backend = False
        self._sync_dist_logs = _sync_dist_logs_enabled(cfg)
        loss_cfg = getattr(cfg, "loss", OmegaConf.create({}))
        self.distortion_loss = DistortionLoss.from_config(getattr(loss_cfg, "distortion", None))
        self.rgb_reconstruction_losses = RGBReconstructionLosses.from_config(getattr(loss_cfg, "rgb", None))
        self.endpoint_rgb_loss = EndpointRGBLoss.from_config(getattr(loss_cfg, "endpoint_rgb", None))

        # Create EMA modules before Lightning checkpoint restore so strict resume
        # can load older checkpoints that include EMA state.
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema_target = None
        if cfg.training.get("ema_target_beta", None) is not None:
            self.ema_target = copy.deepcopy(self.model).eval().requires_grad_(False)

        # Eval Metrics
        self.metrics_list = cfg.training.metrics
        self.match_gt_mean = bool(getattr(cfg.training, "match_gt_mean", False))
        # MetricEval writes temporary image artifacts for FID-like metrics, so each run gets its own
        # scratch directory under /tmp instead of sharing state across launches or ranks.
        self.run_dir = Path(tempfile.mkdtemp(prefix="run_", dir="/tmp"))
        self.metric_evals = None
        self.num_steps_arr = cfg.training.num_steps_arr
        train_metrics_cfg = getattr(cfg.training, "train_metrics", OmegaConf.create({}))
        default_train_metrics_enabled = False
        self.train_metrics_enabled = bool(getattr(train_metrics_cfg, "enabled", default_train_metrics_enabled))
        self.train_metrics_list = list(getattr(train_metrics_cfg, "metrics", ["psnr"]))
        self.train_metrics_num_steps_arr = list(getattr(train_metrics_cfg, "num_steps_arr", self.num_steps_arr))
        self.train_metrics_max_batches_per_epoch = int(getattr(train_metrics_cfg, "max_batches_per_epoch", 1))
        self.train_metrics_every_n_epochs = max(1, int(getattr(train_metrics_cfg, "every_n_epochs", 1)))
        self.train_metric_evals = None
        self._last_train_metric_values = {}
        self._train_metric_finalized_epoch = None
        self._ddp_debug = _ddp_debug_enabled(cfg)
        self._train_step_entry_count = 0
        self._last_train_total_loss = None
        self._encoder_warmup_done = False
        validation_cfg = getattr(cfg, "validation", OmegaConf.create({}))
        self._validation_sampler = _normalize_validation_sampler(
            getattr(validation_cfg, "primary_sampler", getattr(validation_cfg, "sampler", "fm_only"))
        )
        self._validation_samplers = _validation_sampler_list(validation_cfg, self._validation_sampler)
        self._primary_validation_sampler = self._validation_sampler

    def setup(self, stage):

        if self.metric_evals is None:
            self.metric_evals = {}
            metric_names = list(self.metrics_list)
            for metric_name in [
                "global_mean_signed_err",
                "channel_mean_signed_err",
                "r_mean_signed_err",
                "g_mean_signed_err",
                "b_mean_signed_err",
            ]:
                if metric_name not in metric_names:
                    metric_names.append(metric_name)
            for sampler_name in self._validation_samplers:
                self.metric_evals[sampler_name] = {}
                for num_steps in self.num_steps_arr:
                    out_dir = self.run_dir / f"{sampler_name}_sampler" / f"{num_steps}_steps"
                    self.metric_evals[sampler_name][num_steps] = [
                        MetricEval(metric, self.device, out_dir) for metric in metric_names
                    ]
        if self.train_metrics_enabled and self.train_metric_evals is None:
            self.train_metric_evals = {}
            for num_steps in self.train_metrics_num_steps_arr:
                out_dir = self.run_dir / "train" / f"{self._validation_sampler}_sampler" / f"{num_steps}_steps"
                self.train_metric_evals[num_steps] = [
                    MetricEval(metric, self.device, out_dir) for metric in self.train_metrics_list
                ]


    def _set_train_dataset_patch_size(self, patch_size: int) -> None:
        train_loader = getattr(self.trainer, "train_dataloader", None)
        dataset = getattr(train_loader, "dataset", None)
        while dataset is not None:
            if hasattr(dataset, "set_patch_size"):
                dataset.set_patch_size(patch_size)
                return
            dataset = getattr(dataset, "dataset", None)


    def forward(self, batch: Dict[str, Any]) -> Any:
        """
        Define how a batch flows through the model in inference mode.
        """
        raise NotImplementedError

    def compute_loss(
        self,
        images_lq,
        images_hq,
        enable_endpoint_rgb_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Return a dict with at least a "loss" key so we can log other components.
        """

        timing = self._ddp_debug and getattr(self, "_train_step_entry_count", 0) <= 3
        t_compute0 = time.perf_counter()
        if not hasattr(self, "endpoint_rgb_loss"):
            self.endpoint_rgb_loss = EndpointRGBLoss()

        # Augmentation if needed
        t, s, s_prime = self.sample_ts(images_lq)
        if enable_endpoint_rgb_loss and self.endpoint_rgb_loss.time_sampling_active:
            s_prime = self._apply_endpoint_time_sampling_bias(s_prime)
        if timing:
            _sync_cuda_for_debug(images_lq.device)
            t_sample = time.perf_counter()

        beta_scale = torch.ones_like(t)

        if self.cfg.get("debug", {}).get("freeze_noise", False) and self.cfg.get("debug", {}).get("overfit_test", False):
            if not hasattr(self, "_fixed_train_noise"):
                g = torch.Generator(device=images_lq.device)
                g.manual_seed(1234)
                self._fixed_train_noise = torch.randn(images_lq.shape, device=images_lq.device, generator=g)

            noise = self._fixed_train_noise * self.cfg.ir.sigma_lq
        else:
            noise = torch.randn_like(images_lq) * self.cfg.ir.sigma_lq

        prior = images_lq + noise
        yt = (1 - t) * images_hq + t * prior
        ys_prime = (1 - s_prime) * images_hq + s_prime * prior

        v = prior - images_hq

        alpha_value = self.alpha_factor_value()
        alpha = images_lq.new_tensor(alpha_value)
        labels = None
        use_time_gap = bool(getattr(self.cfg.loss, "use_time_gap", False))
        teacher_model = self.ema_target if self.ema_target is not None else self.model
        rgb_aux_active = bool(getattr(getattr(self, "rgb_reconstruction_losses", None), "active", False))
        current_step = getattr(self, "global_step", 0)
        endpoint_rgb_scale = self.endpoint_rgb_loss.schedule_scale(current_step, images_lq)
        endpoint_rgb_active = (
            enable_endpoint_rgb_loss
            and self.endpoint_rgb_loss.active
            and bool(endpoint_rgb_scale.item() > 0.0)
        )
        endpoint_rgb_apply = self._endpoint_rgb_should_apply(endpoint_rgb_active, images_lq)
        loss, logs = self.model(
            yt,
            ys_prime,
            t,
            s,
            s_prime,
            labels,
            beta_scale,
            v,
            teacher_model=teacher_model,
            alpha_factor=alpha_value,
            use_time_gap=use_time_gap,
            lq_latent_cond=images_lq,
            return_aux=self.distortion_loss.active or rgb_aux_active,
        )
        if timing:
            _sync_cuda_for_debug(images_lq.device)
            t_model = time.perf_counter()
        pred_velocity = None
        if self.distortion_loss.active or rgb_aux_active:
            pred_velocity = logs.pop("_pred_velocity", None)
            if pred_velocity is None:
                raise RuntimeError("auxiliary losses require model logs to include '_pred_velocity'")
        if self.distortion_loss.active:
            distortion_loss = self.distortion_loss.from_flow(ys_prime, s_prime, pred_velocity, images_hq)
            if distortion_loss.ndim > 1:
                distortion_loss = distortion_loss.flatten(1).mean(dim=1)
            loss = loss + distortion_loss
            logs["distortion_loss"] = distortion_loss.detach()
        if rgb_aux_active:
            s_prime_b = self.distortion_loss._broadcast_time(s_prime, ys_prime)
            logs["_x0_pred_latent"] = ys_prime.float() - s_prime_b.float() * pred_velocity.float()
        if self.endpoint_rgb_loss.active:
            logs["endpoint_rgb_loss_scale"] = endpoint_rgb_scale.detach().expand(images_lq.shape[0])
            logs["endpoint_rgb_applied"] = images_lq.new_full(
                (images_lq.shape[0],),
                1.0 if endpoint_rgb_apply else 0.0,
                dtype=torch.float32,
            )
        if endpoint_rgb_apply:
            logs["_endpoint_x0_pred_latent"] = self._predict_endpoint_x0_latent(
                prior=prior,
                beta_scale=beta_scale,
                use_time_gap=use_time_gap,
                lq_latent_cond=images_lq,
            )
        if timing:
            _sync_cuda_for_debug(images_lq.device)
            t_distortion = time.perf_counter()
            print(
                f"[ddp-debug train_jvp loss-timing] rank={self.trainer.global_rank} "
                f"step_entries={self._train_step_entry_count} "
                f"sample_ms={(t_sample - t_compute0) * 1000.0:.1f} "
                f"model_forward_ms={(t_model - t_sample) * 1000.0:.1f} "
                f"distortion_ms={(t_distortion - t_model) * 1000.0:.1f} "
                f"alpha_factor={alpha_value:.6g} loss_device={loss.device}",
                flush=True,
            )
        logs["alpha_factor"] = alpha.detach()

        # Check NaN case
        if loss.isnan().any():
            print(f'nan loss {loss.shape} at {logs["ts"]}')
            if bool(getattr(self.cfg.training, "fail_on_nonfinite_loss", False)):
                finite_logs = {}
                for name, value in logs.items():
                    if torch.is_tensor(value):
                        finite_logs[name] = bool(torch.isfinite(value).all().detach().cpu())
                raise RuntimeError(f"Non-finite training loss detected; finite_logs={finite_logs}")
            loss = loss.nan_to_num(nan=0.0)

        return loss, logs

    def _endpoint_rgb_should_apply(self, active: bool, reference: torch.Tensor) -> bool:
        if not active:
            return False
        prob = float(self.endpoint_rgb_loss.prob)
        if prob >= 1.0:
            return True
        if prob <= 0.0:
            return False

        apply_flag = reference.new_empty(())
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                apply_flag.bernoulli_(prob)
            else:
                apply_flag.zero_()
            dist.broadcast(apply_flag, src=0)
        else:
            apply_flag.bernoulli_(prob)
        return bool(apply_flag.item() > 0.5)

    def _apply_endpoint_time_sampling_bias(self, s_prime: torch.Tensor) -> torch.Tensor:
        cfg = self.endpoint_rgb_loss
        out = s_prime
        if cfg.bias_prob > 0.0:
            bias_mask = torch.rand_like(s_prime) < cfg.bias_prob
            biased = cfg.bias_min + torch.rand_like(s_prime) * (1.0 - cfg.bias_min)
            out = torch.where(bias_mask, biased, out)
        if cfg.force_endpoint_prob > 0.0:
            endpoint_mask = torch.rand_like(s_prime) < cfg.force_endpoint_prob
            out = torch.where(endpoint_mask, torch.ones_like(out), out)
        return out

    def _predict_endpoint_x0_latent(
        self,
        *,
        prior: torch.Tensor,
        beta_scale: torch.Tensor,
        use_time_gap: bool,
        lq_latent_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        endpoint_t = torch.ones(prior.shape[0], 1, 1, 1, device=prior.device, dtype=prior.dtype)
        endpoint_s = endpoint_t
        endpoint_velocity = self.model.forward_model(
            prior.to(torch.float32),
            endpoint_t.to(torch.float32),
            endpoint_s.to(torch.float32),
            class_labels=None,
            beta_scale=beta_scale.to(torch.float32),
            use_time_gap=use_time_gap,
            lq_latent_cond=lq_latent_cond,
        )
        return prior.float() - endpoint_velocity.float()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:

        img_lq, img_hq = batch
        self._train_step_entry_count += 1
        debug_this_step = self._ddp_debug and self._train_step_entry_count <= 3
        if debug_this_step:
            root_device = getattr(getattr(self.trainer, "strategy", None), "root_device", self.device)
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            param_device = next(self.model.parameters()).device
            encoder_vae = getattr(self.encoder, "_vae", None)
            encoder_device = next(encoder_vae.parameters()).device if encoder_vae is not None else None
            sampler = getattr(getattr(self.trainer, "train_dataloader", None), "sampler", None)
            _sync_cuda_for_debug(img_lq.device)
            print(
                f"[ddp-debug train_jvp] rank={self.trainer.global_rank} "
                f"local_rank={getattr(self.trainer, 'local_rank', os.environ.get('LOCAL_RANK', 'NA'))} "
                f"step_entries={self._train_step_entry_count} batch_idx={batch_idx} "
                f"root_device={root_device} cuda_current={current_device} "
                f"batch_device={img_lq.device} param_device={param_device} "
                f"batch_shape={tuple(img_lq.shape)} "
                f"encoder_device={encoder_device} sampler={type(sampler).__name__ if sampler else None}",
                flush=True,
            )

        # Convert images to latent space. The VAE needs multiples of 8, and
        # HDiT's patch input needs the resulting latent axes to be even.
        t0 = time.perf_counter()
        img_lq_p, _ = pad_image_if_needed(img_lq, mult=16)
        img_hq_p, meta_img_hq = pad_image_if_needed(img_hq, mult=16)
        images = torch.cat([img_lq_p, img_hq_p], dim=0)
        images = self.encoder.encode(images)
        if debug_this_step:
            _sync_cuda_for_debug(images.device)
        t1 = time.perf_counter()
        img_latent_lq, img_latent_hq = torch.chunk(images, 2, dim=0)

        # Calc Loss (includes the model forward)
        loss, logs = self.compute_loss(img_latent_lq, img_latent_hq)
        x0_pred_latent = logs.pop("_x0_pred_latent", None)
        endpoint_x0_pred_latent = logs.pop("_endpoint_x0_pred_latent", None)
        teacher_forced_rgb_metric_active = self.rgb_reconstruction_losses.active
        if teacher_forced_rgb_metric_active:
            if x0_pred_latent is None:
                raise RuntimeError("RGB endpoint metrics require '_x0_pred_latent'")
            x0_pred_rgb = torch.clamp(self.encoder.decode_float(x0_pred_latent).float(), 0.0, 1.0)
            x0_pred_rgb = unpad_image(x0_pred_rgb, meta_img_hq)
            img_hq_rgb = torch.clamp(img_hq.float() / 255.0, 0.0, 1.0)
            with torch.no_grad():
                teacher_psnr = compute_rgb_psnr(x0_pred_rgb, img_hq_rgb).detach()
                logs["x0_pred_rgb_psnr"] = teacher_psnr
                logs["x0_pred_rgb_psnr_teacher_forced"] = teacher_psnr
            if self.rgb_reconstruction_losses.active:
                rgb_loss, rgb_logs = self.rgb_reconstruction_losses(x0_pred_rgb, img_hq_rgb)
                rgb_loss_scale = self.rgb_reconstruction_losses.schedule_scale(self.global_step, rgb_loss)
                logs["rgb_loss_scale"] = rgb_loss_scale.detach().expand_as(rgb_loss)
                logs["rgb_weight_factor"] = rgb_loss_scale.detach().expand_as(rgb_loss)
                logs["rgb_reconstruction_loss_unscaled"] = rgb_loss.detach()
                logs["rgb_reconstruction_loss"] = (rgb_loss * rgb_loss_scale).detach()
                loss = loss + rgb_loss * rgb_loss_scale
                logs.update(rgb_logs)
        if endpoint_x0_pred_latent is not None:
            endpoint_x0_rgb = torch.clamp(self.encoder.decode_float(endpoint_x0_pred_latent).float(), 0.0, 1.0)
            endpoint_x0_rgb = unpad_image(endpoint_x0_rgb, meta_img_hq)
            img_hq_rgb = torch.clamp(img_hq.float() / 255.0, 0.0, 1.0)
            with torch.no_grad():
                logs["endpoint_x0_rgb_psnr"] = compute_rgb_psnr(endpoint_x0_rgb, img_hq_rgb).detach()
            endpoint_loss = self.endpoint_rgb_loss(endpoint_x0_rgb, img_hq_rgb)
            endpoint_loss_scale = self.endpoint_rgb_loss.schedule_scale(self.global_step, endpoint_loss)
            logs["endpoint_rgb_loss_unscaled"] = endpoint_loss.detach()
            logs["endpoint_rgb_loss"] = (endpoint_loss * endpoint_loss_scale).detach()
            loss = loss + endpoint_loss * endpoint_loss_scale
        loss = loss.mean()
        if debug_this_step:
            _sync_cuda_for_debug(loss.device)
        t2 = time.perf_counter()

        if debug_this_step:
            alpha_log = logs.get("alpha_factor", loss.new_tensor(float("nan")))
            print(
                f"[ddp-debug train_jvp timing] rank={self.trainer.global_rank} "
                f"step_entries={self._train_step_entry_count} batch_idx={batch_idx} "
                f"encode_ms={(t1 - t0) * 1000.0:.1f} "
                f"loss_forward_ms={(t2 - t1) * 1000.0:.1f} "
                f"latent_device={img_latent_lq.device} loss_device={loss.device} "
                f"alpha_factor={float(alpha_log.detach().cpu()):.6g}",
                flush=True,
            )

        # log everything with "train/" prefix
        logs.pop('ts', None)
        log_dict = {f"Train/{k}": v.mean() for k, v in logs.items()}
        log_dict['total_loss'] = loss
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=self._log_to_backend,
            sync_dist=self._sync_dist_logs,
            add_dataloader_idx=False,
        )
        self._last_train_total_loss = float(loss.detach().cpu())
        self._maybe_compute_train_metrics(img_latent_lq, img_hq, meta_img_hq, batch_idx)
        return loss

    def _train_metrics_due_this_epoch(self) -> bool:
        epoch_num = int(self.current_epoch) + 1
        return (epoch_num % self.train_metrics_every_n_epochs) == 0

    def _maybe_compute_train_metrics(self, img_latent_lq, img_hq, meta_img_hq, batch_idx: int) -> None:
        if (
            not self.train_metrics_enabled
            or self.train_metric_evals is None
            or self.train_metrics_max_batches_per_epoch <= 0
            or batch_idx >= self.train_metrics_max_batches_per_epoch
            or not self._train_metrics_due_this_epoch()
        ):
            return

        if not hasattr(self, "_fixed_train_noise_cache"):
            self._fixed_train_noise_cache = {}
        key = (tuple(img_latent_lq.shape), img_latent_lq.device.type, img_latent_lq.device.index, img_latent_lq.dtype)
        noise = self._fixed_train_noise_cache.get(key, None)
        if noise is None or noise.device != img_latent_lq.device or noise.dtype != img_latent_lq.dtype:
            g = torch.Generator(device=img_latent_lq.device)
            g.manual_seed(4321)
            noise = torch.randn(img_latent_lq.shape, device=img_latent_lq.device, dtype=img_latent_lq.dtype, generator=g)
            noise = noise * self.cfg.ir.sigma_lq
            self._fixed_train_noise_cache[key] = noise

        bs = img_latent_lq.shape[0]
        prior = img_latent_lq.detach() + noise[:bs]
        sampler = self._validation_sampler
        gen_fn = generator_fn if sampler == "jump" else generator_fn_fm_only
        use_time_gap = bool(getattr(self.cfg.loss, "use_time_gap", False))
        lq_latent_cond = img_latent_lq.detach() if "LQ_Cond" in self.cfg.network.model_type else None

        with torch.no_grad():
            for num_steps in self.train_metrics_num_steps_arr:
                pred_latent = gen_fn(
                    self.model,
                    prior,
                    class_labels=None,
                    num_steps=num_steps,
                    cfg_scale=None,
                    lq_latent_cond=lq_latent_cond,
                    use_time_gap=use_time_gap,
                )
                pred = self.encoder.decode(pred_latent)
                pred = unpad_image(pred, meta_img_hq)
                self.compute_train_metrics(pred, img_hq, num_steps=num_steps)

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        img_lq, img_hq = batch
        val_names = list(self.cfg.dataset.get("val_names", []))
        val_name = val_names[dataloader_idx] if dataloader_idx < len(val_names) else None
        val_prefix = f"Val/{val_name}" if val_name else "Val"

        # Build inference latents from LQ only. GT is touched later, after
        # predictions exist, for losses and metrics only.
        img_lq_p, meta_img_lq = pad_image_if_needed(img_lq, mult=8)
        img_hq_p, meta_img_hq = pad_image_if_needed(img_hq, mult=8)
        img_latent_lq = self.encoder.encode(img_lq_p)

        # (HDiT latent padding) mult=32
        B, C, H, W = img_latent_lq.shape
        multiple = 32
        if needs_padding(H, W, multiple):
            img_latent_lq, meta_latent = pad_latent_to_multiple(img_latent_lq, mult=multiple)
        else:
            meta_latent = None

        # Generate Images
        # deterministic noise cache per shape
        if not hasattr(self, "_fixed_val_noise_cache"):
            self._fixed_val_noise_cache = {}

        key = (tuple(img_latent_lq.shape), img_latent_lq.device.type, img_latent_lq.device.index, img_latent_lq.dtype)

        noise = self._fixed_val_noise_cache.get(key, None)
        if noise is None or noise.device != img_latent_lq.device or noise.dtype != img_latent_lq.dtype:
            g = torch.Generator(device=img_latent_lq.device)
            g.manual_seed(1234)
            noise = torch.randn(img_latent_lq.shape, device=img_latent_lq.device, dtype=img_latent_lq.dtype,
                                generator=g)
            noise = noise * self.cfg.ir.sigma_lq
            self._fixed_val_noise_cache[key] = noise

        bs = img_latent_lq.shape[0]
        prior = img_latent_lq + noise[:bs]

        if 'LQ_Cond' in self.cfg.network.model_type:
            lq_latent_cond = img_latent_lq
        else:
            lq_latent_cond = None
        include_num_steps = len(self.num_steps_arr) > 1
        res_imgs_by_sampler = {}
        use_time_gap = bool(getattr(self.cfg.loss, "use_time_gap", False))
        endpoint_psnr = None
        if self.endpoint_rgb_loss.enabled:
            endpoint_x0_latent = self._predict_endpoint_x0_latent(
                prior=prior,
                beta_scale=torch.ones(bs, 1, 1, 1, device=prior.device, dtype=prior.dtype),
                use_time_gap=use_time_gap,
                lq_latent_cond=lq_latent_cond,
            )
            if meta_latent is not None:
                endpoint_x0_latent = unpad_latent(endpoint_x0_latent, meta_latent)
            endpoint_x0_rgb = torch.clamp(self.encoder.decode_float(endpoint_x0_latent).float(), 0.0, 1.0)
            endpoint_gt_rgb = torch.clamp(img_hq_p.float() / 255.0, 0.0, 1.0)
            endpoint_psnr = compute_rgb_psnr(endpoint_x0_rgb, endpoint_gt_rgb).mean()
            endpoint_logs = (
                {f"{val_name}/endpoint_x0_rgb_psnr": endpoint_psnr}
                if val_name
                else {"endpoint_x0_rgb_psnr": endpoint_psnr}
            )
            if val_name is None or dataloader_idx == 0:
                endpoint_logs["endpoint_x0_rgb_psnr"] = endpoint_psnr
                endpoint_logs["Val/endpoint_x0_rgb_psnr"] = endpoint_psnr
            self.log_dict(
                endpoint_logs,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=self._log_to_backend,
                sync_dist=self._sync_dist_logs,
                add_dataloader_idx=False,
            )
        for sampler in self._validation_samplers:
            gen_fn = generator_fn if sampler == "jump" else generator_fn_fm_only
            is_primary_sampler = sampler == self._validation_sampler
            res_imgs_by_step = {}
            for num_steps in self.num_steps_arr:
                images_hq_latent_estimate = gen_fn(
                    self.model, prior, class_labels=None, num_steps=num_steps, cfg_scale=None, lq_latent_cond=lq_latent_cond, use_time_gap=use_time_gap,
                )
                # (1) Unpad latent back to pre-HDiT latent size
                if meta_latent is not None:
                    images_hq_latent_estimate = unpad_latent(images_hq_latent_estimate, meta_latent)

                images_hq_estimate = self.encoder.decode(images_hq_latent_estimate)
                # Remove image padding (only if applied)
                images_hq_estimate = unpad_image(images_hq_estimate, meta_img_hq)

                self.compute_metrics(images_hq_estimate, img_hq, num_steps=num_steps, sampler_name=sampler)
                pred_norm = torch.clamp(images_hq_estimate.float() / 255.0, 0.0, 1.0)
                gt_norm = torch.clamp(img_hq.float() / 255.0, 0.0, 1.0)
                mse = torch.mean((pred_norm - gt_norm) ** 2, dim=(1, 2, 3))
                psnr = (-10.0 * torch.log10(torch.clamp(mse, min=1e-12))).mean()
                mean_error = (pred_norm - gt_norm).mean()
                mean_abs_error = (pred_norm.mean(dim=(2, 3)) - gt_norm.mean(dim=(2, 3))).abs().mean()
                metric_logs = (
                    {
                        f"{val_name}/{sampler}/psnr/num_steps_{num_steps}_epoch": psnr,
                        f"{val_name}/{sampler}/sampled_psnr/num_steps_{num_steps}_epoch": psnr,
                        f"{val_name}/{sampler}/sampled_rgb_psnr_num_steps_{num_steps}": psnr,
                        f"{val_name}/{sampler}/mean_error/num_steps_{num_steps}_epoch": mean_error,
                        f"{val_name}/{sampler}/mean_abs_error/num_steps_{num_steps}_epoch": mean_abs_error,
                    }
                    if val_name
                    else {
                        f"{sampler}/psnr/num_steps_{num_steps}_epoch": psnr,
                        f"{sampler}/sampled_psnr/num_steps_{num_steps}_epoch": psnr,
                        f"{sampler}/sampled_rgb_psnr_num_steps_{num_steps}": psnr,
                        f"{sampler}/mean_error/num_steps_{num_steps}_epoch": mean_error,
                        f"{sampler}/mean_abs_error/num_steps_{num_steps}_epoch": mean_abs_error,
                    }
                )
                if val_name is None or dataloader_idx == 0:
                    metric_logs[f"Val/{sampler}/sampled_rgb_psnr_num_steps_{num_steps}"] = psnr
                if is_primary_sampler:
                    metric_logs.update(
                        (
                            {
                                f"{val_name}/psnr/num_steps_{num_steps}_epoch": psnr,
                                f"{val_name}/sampled_psnr/num_steps_{num_steps}_epoch": psnr,
                                f"{val_name}/sampled_rgb_psnr_num_steps_{num_steps}": psnr,
                                f"{val_name}/mean_error/num_steps_{num_steps}_epoch": mean_error,
                                f"{val_name}/mean_abs_error/num_steps_{num_steps}_epoch": mean_abs_error,
                            }
                            if val_name
                            else {
                                f"psnr/num_steps_{num_steps}_epoch": psnr,
                                f"sampled_psnr/num_steps_{num_steps}_epoch": psnr,
                                f"sampled_rgb_psnr_num_steps_{num_steps}": psnr,
                                f"mean_error/num_steps_{num_steps}": mean_error,
                                f"mean_abs_error/num_steps_{num_steps}": mean_abs_error,
                                f"mean_error/num_steps_{num_steps}_epoch": mean_error,
                                f"mean_abs_error/num_steps_{num_steps}_epoch": mean_abs_error,
                            }
                        )
                    )
                    if val_name is None or dataloader_idx == 0:
                        metric_logs[f"Val/sampled_rgb_psnr_num_steps_{num_steps}"] = psnr
                metric_logs.update(
                    {
                        _validation_metric_name(
                            metric_name,
                            sampler,
                            num_steps=num_steps,
                            include_num_steps=include_num_steps,
                            val_name=val_name,
                        ): metric_value
                        for metric_name, metric_value in {
                            "psnr": psnr,
                            "mean_error": mean_error,
                            "mean_abs_error": mean_abs_error,
                            "sampled_rgb_psnr": psnr,
                        }.items()
                    }
                )
                self.log_dict(
                    metric_logs,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=self._log_to_backend,
                    sync_dist=self._sync_dist_logs,
                    add_dataloader_idx=False,
                )
                res_imgs_by_step[num_steps] = {
                    "img_lq": img_lq.detach().cpu().numpy(),
                    "img_hq": img_hq.detach().cpu().numpy(),
                    "img_hq_estimate": images_hq_estimate.detach().cpu().numpy(),
                }

            res_imgs_by_sampler[sampler] = res_imgs_by_step

        # Calc Val Loss after predictions so GT cannot influence inference.
        img_hq_p, meta_img_hq = pad_image_if_needed(img_hq, mult=8)
        img_latent_hq = self.encoder.encode(img_hq_p)
        if meta_latent is not None:
            img_latent_hq, _ = pad_latent_to_multiple(img_latent_hq, mult=multiple)
        loss, logs = self.compute_loss(
            img_latent_lq,
            img_latent_hq,
            enable_endpoint_rgb_loss=False,
        )
        x0_pred_latent = logs.pop("_x0_pred_latent", None)
        logs.pop("_endpoint_x0_pred_latent", None)
        if self.rgb_reconstruction_losses.active:
            if x0_pred_latent is None:
                raise RuntimeError("RGB reconstruction losses require '_x0_pred_latent'")
            if meta_latent is not None:
                x0_pred_latent = unpad_latent(x0_pred_latent, meta_latent)
            x0_pred_rgb = torch.clamp(self.encoder.decode_float(x0_pred_latent).float(), 0.0, 1.0)
            img_hq_rgb = torch.clamp(img_hq_p.float() / 255.0, 0.0, 1.0)
            with torch.no_grad():
                teacher_psnr = compute_rgb_psnr(x0_pred_rgb, img_hq_rgb).detach()
                logs["x0_pred_rgb_psnr"] = teacher_psnr
                logs["x0_pred_rgb_psnr_teacher_forced"] = teacher_psnr
            rgb_loss, rgb_logs = self.rgb_reconstruction_losses(x0_pred_rgb, img_hq_rgb)
            rgb_loss_scale = self.rgb_reconstruction_losses.schedule_scale(self.global_step, rgb_loss)
            logs["rgb_loss_scale"] = rgb_loss_scale.detach().expand_as(rgb_loss)
            logs["rgb_weight_factor"] = rgb_loss_scale.detach().expand_as(rgb_loss)
            logs["rgb_reconstruction_loss_unscaled"] = rgb_loss.detach()
            logs["rgb_reconstruction_loss"] = (rgb_loss * rgb_loss_scale).detach()
            loss = loss + rgb_loss * rgb_loss_scale
            logs.update(rgb_logs)
        loss = loss.mean()

        logs.pop('ts', None)
        logs['total_loss'] = loss
        log_dict = {f"{val_prefix}/{k}": v.mean() for k, v in logs.items()}
        if val_name is None or dataloader_idx == 0:
            log_dict.update({f"Val/{k}": v.mean() for k, v in logs.items()})

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=self._log_to_backend,
            sync_dist=self._sync_dist_logs,
        )
        for sampler in self._validation_samplers:
            sampler_loss_logs = {
                _validation_metric_name(metric_name, sampler, val_name=val_name): value.mean()
                for metric_name, value in logs.items()
            }
            if endpoint_psnr is not None:
                sampler_loss_logs[_validation_metric_name("endpoint_x0_rgb_psnr", sampler, val_name=val_name)] = endpoint_psnr
            self.log_dict(
                sampler_loss_logs,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=self._log_to_backend,
                sync_dist=self._sync_dist_logs,
                add_dataloader_idx=False,
            )

        return loss

    def configure_optimizers(self):
        """
        Create optimizer (and scheduler) using the TVM YAML config.
        """

        # Example: mimic typical configs with optimizer.{lr, beta1, beta2, weight_decay}
        optimizer_kwargs = self.cfg.optimizer
        optimizer = resolver.instantiate(params=self.model.parameters(), **optimizer_kwargs)

        # --- Optional LR scheduler controlled by Hydra config ---
        sched_cfg = self.cfg.get("scheduler", None)
        if sched_cfg is None or not sched_cfg.get("enable", False):
            return optimizer

        warmup_steps = int(sched_cfg.get("warmup_steps", 0))
        total_steps = int(sched_cfg.get("total_steps", 50000))
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

        lr_sched = build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_sched,
                "interval": "step",  # IMPORTANT: per-step schedule
                "frequency": 1,
                "name": "lr",
            },
        }

    def sample_ts(self, images ):

        if self.cfg.get("debug", {}).get("freeze_ts", False) and self.cfg.get("debug", {}).get("overfit_test", False):
            return self.frozen_schedule_ts(images)

        if self.cfg.loss.ts_dist == 'uniform':
            if self.cfg.loss.ts_dist_uniform_type == 'discrete':
                return self.sample_ts_discrete(images)
            elif self.cfg.loss.ts_dist_uniform_type == 'continuous':
                return self.uniform_sample_ts(images)
            elif self.cfg.loss.ts_dist_uniform_type == 'stratified':
                bs = images.shape[0]
                dtype = images.dtype
                return self.stratified_uniform(bs, self.trainer.global_rank, self.trainer.world_size, dtype=dtype,
                                        device=images.device)
            else:
                raise NotImplementedError
        elif self.cfg.loss.ts_dist == 'uniform_t_and_s':
            bs = images.shape[0]
            dtype = images.dtype
            return self.uniform_t_and_s(bs, dtype=dtype, device=images.device)
        else:
            return self.org_sample_ts(images)

    def org_sample_ts(self, images ):
        log_len = torch.randn([images.shape[0], 1, 1, 1],
                              device=images.device) * self.cfg.loss.gap_P_std + self.cfg.loss.gap_P_mean
        len_ = log_len.sigmoid()
        maxs = 1 - len_
        max_logns = torch.log(maxs / (1 - maxs))

        dist = torch.distributions.Normal(
            loc=torch.full_like(len_, self.cfg.loss.s_P_mean),
            scale=torch.full_like(len_, self.cfg.loss.s_P_std),
        )

        cdf = torch.rand_like(len_) * (
                dist.cdf(max_logns) - dist.cdf(torch.full_like(len_, -np.inf))
        ) + dist.cdf(torch.full_like(len_, -np.inf))
        log_ns = dist.icdf(cdf)
        s = log_ns.sigmoid()

        t = (s + len_).clamp(max=1)

        log_ns_prime = torch.randn_like(len_) * self.cfg.loss.s_P_std + self.cfg.loss.s_P_mean
        s_prime = log_ns_prime.sigmoid()

        return t, s, s_prime

    def uniform_sample_ts(self, images ):

        # Sample t Uniform(0, 1) elementwise
        t = torch.rand([images.shape[0], 1, 1, 1], device=images.device)

        # s ~ Uniform(0, t) elementwise
        s = torch.rand_like(t) * t

        # independent s_prime ~ Uniform(0, 1)
        s_prime = torch.rand_like(t)

        return t, s, s_prime

    def uniform_t_and_s(self, bs, dtype=None, device=None):
        """
        Non-stratified sampling of (t, s, s_prime) with:
          - t, s in [0,1], t > s
          - s_prime ~ Uniform[0,1]
        Output shape: (bs, 1, 1, 1)
        """
        u = torch.rand(bs, 2, dtype=dtype, device=device)
        u_p = torch.rand(bs, dtype=dtype, device=device)

        t = torch.maximum(u[:, 0], u[:, 1])
        s = torch.minimum(u[:, 0], u[:, 1])

        t = t.view(bs, 1, 1, 1)
        s = s.view(bs, 1, 1, 1)
        s_prime = u_p.view(bs, 1, 1, 1)

        return t, s, s_prime

    def frozen_schedule_ts(self, images):
        B = images.shape[0]
        device = images.device

        N = int(self.cfg.get("debug", {}).get("freeze_steps", 4))

        grid = torch.linspace(1.0, 0.0, steps=N + 1, device=device)
        mids = 0.5 * (grid[:-1] + grid[1:])

        idx = int(self.global_step % N)
        t_val = mids[idx]

        t = torch.full((B, 1, 1, 1), float(t_val), device=device)
        s = 0.5 * t
        s_prime = t

        return t, s, s_prime

    def sample_ts_discrete(self, images, num_steps=25):
        """
        Discrete time sampling matched to N-step Euler inference.

        - Samples t from midpoints of linspace(1 -> 0)
        - Uses s = t (single-time FM)
        """

        B = images.shape[0]
        device = images.device
        dtype = images.dtype

        t_nodes = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        t_mids = 0.5 * (t_nodes[:-1] + t_nodes[1:])

        if getattr(self.cfg.training, "cycle_t", False):
            idx = int(self.global_step % num_steps)
            t_val = t_mids[idx]
        else:
            idx = torch.randint(0, num_steps, (1,), device=device)
            t_val = t_mids[idx]

        t = t_val.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        s = t
        s_prime = t

        return t, s, s_prime

    def stratified_uniform(self, bs, group=0, groups=1, dtype=None, device=None):
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if group < 0 or group >= groups:
            raise ValueError(f"group must be in [0, {groups})")
        n = bs * groups
        offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
        u = torch.rand(bs, dtype=dtype, device=device)
        t = ((offsets + u) / n).view(bs, 1, 1, 1)
        s = t
        s_prime = t
        return t, s, s_prime

    @torch.no_grad()
    def on_before_zero_grad(self, optimizer) -> None:
        ema_beta = self.cfg.training.ema_beta
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        if self.ema_target is not None:
            ema_target_beta = self.cfg.training.ema_target_beta
            for p_ema, p_net in zip(self.ema_target.parameters(), self.model.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_target_beta))

    def on_fit_start(self):

        # Seed things
        base_seed = int(self.cfg.training.seed)

        rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        root_device = getattr(getattr(self.trainer, "strategy", None), "root_device", self.device)
        if torch.cuda.is_available() and isinstance(root_device, torch.device) and root_device.type == "cuda":
            torch.cuda.set_device(_cuda_device_index(root_device))

        np_seed = (base_seed * world_size + rank) % (1 << 31)
        np.random.seed(np_seed)

        torch_seed = int(np.random.randint(1 << 31))
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

        if self._ddp_debug:
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            param_device = next(self.model.parameters()).device
            print(
                f"[ddp-debug train_jvp] rank={rank} "
                f"local_rank={getattr(self.trainer, 'local_rank', os.environ.get('LOCAL_RANK', 'NA'))} "
                f"world_size={world_size} root_device={root_device} "
                f"cuda_current={current_device} param_device={param_device}",
                flush=True,
            )
        if dist.is_available() and dist.is_initialized():
            # 1) Ensure all ranks reached here
            _dist_barrier(root_device)

            # 2) Model should already be on GPU here for NCCL broadcast
            assert next(self.model.parameters()).is_cuda, "Model params are not on CUDA at on_fit_start"

            # 3) Broadcast rank0's initialized weights to all ranks
            _broadcast_module(self.model, src=0)
            _dist_barrier(root_device)

        # Create EMA AFTER model is synced (on all ranks)
        if self.ema is None:
            self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)

        if self.cfg.training.get("ema_target_beta", None) is not None and self.ema_target is None:
            self.ema_target = copy.deepcopy(self.model).eval().requires_grad_(False)

        # Optional: broadcast EMA too (safe + keeps everyone identical)
        if dist.is_available() and dist.is_initialized():
            _broadcast_module(self.ema, src=0)
            if self.ema_target is not None:
                _broadcast_module(self.ema_target, src=0)
            _dist_barrier(root_device)

        self._warmup_encoder(root_device)

        if self._ddp_debug:
            self.print(
                f"[rank {self.trainer.global_rank}] EMA ready: "
                f"ema={self.ema is not None}, ema_target={self.ema_target is not None}"
            )

    @torch.no_grad()
    def _warmup_encoder(self, root_device: torch.device):
        if self._encoder_warmup_done:
            return
        debug_cfg = getattr(self.cfg, "debug", OmegaConf.create({}))
        if not bool(getattr(debug_cfg, "encoder_warmup", True)):
            return
        if not isinstance(root_device, torch.device):
            return
        if not hasattr(self.encoder, "init"):
            return

        t0 = time.perf_counter()
        # Some encoders lazily materialize heavy submodules on first use; doing it here keeps that
        # one-time cost out of the first real optimization step and makes DDP startup more uniform.
        self.encoder.init(root_device)
        _sync_cuda_for_debug(root_device)
        init_ms = (time.perf_counter() - t0) * 1000.0

        did_dummy = False
        dummy_ms = 0.0
        input_res = int(getattr(self.cfg.ir, "input_res", 0))
        max_warmup_res = int(getattr(debug_cfg, "encoder_warmup_max_res", 512))
        if input_res > 0 and input_res <= max_warmup_res:
            loader = getattr(self.trainer, "train_dataloader", None)
            loader_bs = getattr(loader, "batch_size", None) or getattr(self.cfg.training, "batch_gpu", 1) or 1
            warmup_bs = int(getattr(debug_cfg, "encoder_warmup_batch", 2 * int(loader_bs)))
            warmup_bs = max(1, warmup_bs)
            dummy = torch.zeros(
                warmup_bs,
                3,
                input_res,
                input_res,
                device=root_device,
                dtype=torch.float32,
            )
            t1 = time.perf_counter()
            _ = self.encoder.encode(dummy)
            _sync_cuda_for_debug(root_device)
            dummy_ms = (time.perf_counter() - t1) * 1000.0
            did_dummy = True

        self._encoder_warmup_done = True
        if dist.is_available() and dist.is_initialized():
            _dist_barrier(root_device)

        if self._ddp_debug:
            encoder_vae = getattr(self.encoder, "_vae", None)
            encoder_device = next(encoder_vae.parameters()).device if encoder_vae is not None else None
            print(
                f"[ddp-debug train_jvp encoder-warmup] rank={self.trainer.global_rank} "
                f"root_device={root_device} encoder_device={encoder_device} "
                f"init_ms={init_ms:.1f} dummy={did_dummy} dummy_ms={dummy_ms:.1f}",
                flush=True,
            )


    def alpha_factor_value(self) -> float:
        """
        Returns the current alpha factor as a Python float.
        Schedule:
          - warmup_epochs: alpha=0
          - then either:
              mode="step" or "constant": alpha=alpha_final
              mode="ramp": linear ramp over ramp_steps to alpha_final
              mode="stair_hold": held stair steps capped by alpha_final
        """
        cfg = self.cfg.loss_weighting
        alpha_final = float(cfg.alpha_final)
        warmup_epochs = int(cfg.warmup_epochs)
        warmup_steps = int(getattr(cfg, "warmup_steps", 0))
        ramp_steps = int(getattr(cfg, "ramp_steps", 0))
        mode = str(getattr(cfg, "mode", "ramp")).lower()

        if self.current_epoch < warmup_epochs or self.global_step < warmup_steps:
            a = 0.0
        else:
            if mode in ("step", "constant"):
                a = alpha_final
            elif mode == "ramp":
                if ramp_steps <= 0:
                    a = alpha_final
                else:
                    # steps since warmup ended
                    # Note: global_step is per-optimizer-step (after grad accumulation)
                    warmup_end_global_step = max(
                        getattr(self, "_warmup_end_global_step", 0),
                        warmup_steps,
                    )
                    steps_since = max(0, self.global_step - warmup_end_global_step)
                    frac = min(1.0, steps_since / float(ramp_steps))
                    a = frac * alpha_final
            elif mode == "stair_hold":
                if "stair_step_every" not in cfg:
                    raise ValueError("loss_weighting.stair_step_every is required for mode='stair_hold'")
                if "stair_step_size" not in cfg:
                    raise ValueError("loss_weighting.stair_step_size is required for mode='stair_hold'")
                stair_step_every = int(cfg.stair_step_every)
                stair_step_size = float(cfg.stair_step_size)
                if stair_step_every <= 0:
                    raise ValueError("loss_weighting.stair_step_every must be > 0 for mode='stair_hold'")
                if stair_step_size <= 0.0:
                    raise ValueError("loss_weighting.stair_step_size must be > 0 for mode='stair_hold'")
                # steps since warmup ended
                # Note: global_step is per-optimizer-step (after grad accumulation)
                steps_since = max(0, self.global_step - warmup_steps)
                stair_index = steps_since // stair_step_every
                a = min(alpha_final, stair_step_size * float(stair_index + 1))
            else:
                raise ValueError(f"Unknown alpha schedule mode: {mode}")

        return float(a)

    def alpha_factor(self) -> torch.Tensor:
        return torch.tensor(self.alpha_factor_value(), device=self.device, dtype=torch.float32)

    def on_train_epoch_start(self):
        train_loader = getattr(self.trainer, "train_dataloader", None)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(int(self.current_epoch))
        # Record the global_step when warmup just ended (first epoch after warmup)
        warmup_epochs = int(self.cfg.loss_weighting.warmup_epochs)
        if self.current_epoch == warmup_epochs and not hasattr(self, "_warmup_end_global_step"):
            self._warmup_end_global_step = int(self.global_step)

    def compute_metrics_ORG(self, x_hq_hat, x_hq, num_steps, sampler_name: Optional[str] = None):

        x_hq_hat = torch.clamp(x_hq_hat.float() / 255.0, min=0.0, max=1.0)  # expect in [0,1]
        x_hq = torch.clamp(x_hq.float() / 255.0, min=0.0, max=1.0)  # expect in [0,1]
        sampler_name = sampler_name or self._primary_validation_sampler

        for metric_eval in self.metric_evals[sampler_name][num_steps]:
            metric_eval.compute(x_hq_hat, x_hq)

    def compute_metrics_debug(self, x_hq_hat, x_hq, num_steps, sampler_name: Optional[str] = None):

        # -------------------------
        # rank / step
        # -------------------------
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        step = int(num_steps)
        sampler_name = sampler_name or self._primary_validation_sampler

        # -------------------------
        # Normalize on GPU
        # IMPORTANT: do NOT rely on .contiguous() alone (it may keep channels_last)
        # -------------------------
        x_hq_hat = torch.clamp(x_hq_hat.float() / 255.0, 0.0, 1.0).detach()
        x_hq = torch.clamp(x_hq.float() / 255.0, 0.0, 1.0).detach()

        # -------------------------
        # CPU snapshot FIRST (safe stats / logging even if CUDA later crashes)
        # -------------------------
        # NOTE: if the device is already "poisoned" by a prior CUDA error, .cpu() can also fail.
        # But doing it before pyiqa greatly increases the chance you capture the tensors.
        x_hat_cpu = x_hq_hat.to("cpu", non_blocking=False)
        x_gt_cpu = x_hq.to("cpu", non_blocking=False)

        # Safe CPU stats
        hat_min = x_hat_cpu.min().item()
        hat_max = x_hat_cpu.max().item()
        gt_min = x_gt_cpu.min().item()
        gt_max = x_gt_cpu.max().item()
        hat_finite = torch.isfinite(x_hat_cpu).all().item()
        gt_finite = torch.isfinite(x_gt_cpu).all().item()

        # -------------------------
        # Iterate metrics
        # -------------------------
        for metric_eval in self.metric_evals[sampler_name][num_steps]:
            name = metric_eval.metric

            try:
                # Make CUDA errors show up on the correct line (helpful during debugging)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                if name in {"lpips", "lpips_alex"}:
                    # ---- Debug print to confirm memory format / strides ----
                    # If you see channels_last=True and contig_format=False, that's the smoking gun.
                    print(
                        f"[rank {rank}] LPIPS input layout: "
                        f"contig={x_hq_hat.is_contiguous()} "
                        f"contig_format={x_hq_hat.is_contiguous(memory_format=torch.contiguous_format)} "
                        f"channels_last={x_hq_hat.is_contiguous(memory_format=torch.channels_last)} "
                        f"stride={x_hq_hat.stride()} dtype={x_hq_hat.dtype} device={x_hq_hat.device}"
                    )

                    # ---- FORCE safe layout + fp32 ----
                    # contiguous_format is key; clone forces a fresh aligned allocation.
                    pred_lp = (
                        x_hq_hat.detach()
                        .to(dtype=torch.float32, memory_format=torch.contiguous_format)
                        .contiguous(memory_format=torch.contiguous_format)
                        .clone()
                    )
                    gt_lp = (
                        x_hq.detach()
                        .to(dtype=torch.float32, memory_format=torch.contiguous_format)
                        .contiguous(memory_format=torch.contiguous_format)
                        .clone()
                    )

                    # ---- Disable autocast for LPIPS ----
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                        metric_eval.compute(pred_lp, gt_lp)

                else:
                    # For other metrics keep as-is (they already work for you)
                    metric_eval.compute(x_hq_hat, x_hq)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            except Exception as e:
                # -------------------------
                # Log + dump for offline inspection
                # -------------------------
                print(f"\n[rank {rank}] ❌ METRIC FAILED: {name}")
                print(f"step={step}")
                print(
                    f"pred: min={hat_min:.6f}, max={hat_max:.6f}, finite={hat_finite}, shape={tuple(x_hat_cpu.shape)}"
                )
                print(
                    f"gt  : min={gt_min:.6f},  max={gt_max:.6f},  finite={gt_finite},  shape={tuple(x_gt_cpu.shape)}"
                )
                print(f"exception: {repr(e)}\n")

                # Write to /tmp (or change to your run dir)
                out_path = f"/tmp/pyiqa_fail_rank{rank}_step{step}_{name}.pt"
                try:
                    torch.save({"pred": x_hat_cpu, "gt": x_gt_cpu}, out_path)
                    print(f"[rank {rank}] Saved failing tensors to: {out_path}")
                except Exception as dump_e:
                    print(f"[rank {rank}] Failed to torch.save tensors: {repr(dump_e)}")

                raise

    def compute_metrics(self, x_hq_hat, x_hq, num_steps, sampler_name: str):


        # -------------------------
        # Normalize on GPU
        # IMPORTANT: do NOT rely on .contiguous() alone (it may keep channels_last)
        # -------------------------
        x_hq_hat = torch.clamp(x_hq_hat.float() / 255.0, 0.0, 1.0).detach()
        x_hq = torch.clamp(x_hq.float() / 255.0, 0.0, 1.0).detach()
        x_hq_hat_gt_mean = None

        # -------------------------
        # Iterate metrics
        # -------------------------
        for metric_eval in self.metric_evals[sampler_name][num_steps]:
            name = metric_eval.metric
            metric_pred = x_hq_hat
            if self.match_gt_mean and name in GT_MEAN_MATCH_METRICS:
                if x_hq_hat_gt_mean is None:
                    x_hq_hat_gt_mean = match_prediction_mean_to_gt(x_hq_hat, x_hq)
                metric_pred = x_hq_hat_gt_mean

            if name in {"lpips", "lpips_alex"}:
                pred_lp = (
                    metric_pred.detach()
                    .to(dtype=torch.float32, memory_format=torch.contiguous_format)
                    .contiguous(memory_format=torch.contiguous_format)
                    .clone()
                )
                gt_lp = (
                    x_hq.detach()
                    .to(dtype=torch.float32, memory_format=torch.contiguous_format)
                    .contiguous(memory_format=torch.contiguous_format)
                    .clone()
                )

                # ---- Disable autocast for LPIPS ----
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    metric_eval.compute(pred_lp, gt_lp)
            else:
                # For other metrics keep as-is (they already work for you)
                metric_eval.compute(metric_pred, x_hq)

    def compute_train_metrics(self, x_hq_hat, x_hq, num_steps):
        x_hq_hat = torch.clamp(x_hq_hat.float() / 255.0, 0.0, 1.0).detach()
        x_hq = torch.clamp(x_hq.float() / 255.0, 0.0, 1.0).detach()
        x_hq_hat_gt_mean = None

        for metric_eval in self.train_metric_evals[num_steps]:
            name = metric_eval.metric
            metric_pred = x_hq_hat
            if self.match_gt_mean and name in GT_MEAN_MATCH_METRICS:
                if x_hq_hat_gt_mean is None:
                    x_hq_hat_gt_mean = match_prediction_mean_to_gt(x_hq_hat, x_hq)
                metric_pred = x_hq_hat_gt_mean

            if name in {"lpips", "lpips_alex"}:
                pred_lp = (
                    metric_pred.detach()
                    .to(dtype=torch.float32, memory_format=torch.contiguous_format)
                    .contiguous(memory_format=torch.contiguous_format)
                    .clone()
                )
                gt_lp = (
                    x_hq.detach()
                    .to(dtype=torch.float32, memory_format=torch.contiguous_format)
                    .contiguous(memory_format=torch.contiguous_format)
                    .clone()
                )
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    metric_eval.compute(pred_lp, gt_lp)
            else:
                metric_eval.compute(metric_pred, x_hq)

    def finalize_train_metric_logs(self):
        if not self.train_metrics_enabled or self.train_metric_evals is None:
            return {}
        if not self._train_metrics_due_this_epoch():
            return {}
        epoch = int(self.current_epoch)
        if self._train_metric_finalized_epoch == epoch:
            return dict(self._last_train_metric_values)

        values = {}
        for num_steps, metric_evals in self.train_metric_evals.items():
            for metric_eval in metric_evals:
                val = metric_eval.get_final()
                metric_log_name = f"Train/{metric_eval.metric}/num_steps_{num_steps}_epoch"
                if self.trainer.is_global_zero:
                    values[metric_log_name] = float(val.detach().cpu()) if torch.is_tensor(val) else float(val)
                    self.log(
                        metric_log_name,
                        val,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                        prog_bar=True,
                        logger=self._log_to_backend,
                    )

        self._last_train_metric_values = values
        self._train_metric_finalized_epoch = epoch
        return dict(values)

    def on_train_epoch_end(self):
        self.finalize_train_metric_logs()


    def on_validation_epoch_end(self):
        include_num_steps = len(self.num_steps_arr) > 1
        for sampler_name, metric_evals_by_step in self.metric_evals.items():
            for num_steps, metric_evals in metric_evals_by_step.items():
                for metric_eval in metric_evals:
                    val = metric_eval.get_final()  # keep as tensor

                    metric_log_name = _validation_metric_name(
                        metric_eval.metric,
                        sampler_name,
                        num_steps=num_steps,
                        include_num_steps=include_num_steps,
                    )

                    # Log only from global rank 0 to keep epoch summaries deterministic.
                    if self.trainer.is_global_zero:
                        self.log(
                            metric_log_name,
                            val,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=False,  # IMPORTANT: don't ask Lightning to sync here
                            prog_bar=True,
                            logger=self._log_to_backend,
                        )
        if self.trainer.is_global_zero:
            self._stdout_epoch_summary()
            print(
                f"TRAIN_JVP_EPOCH_COMPLETE epoch={self.current_epoch + 1}",
                flush=True,
            )

        torch.cuda.empty_cache()

    @staticmethod
    def _scalar_metric_value(value: Any) -> Optional[float]:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().float().cpu().item()
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            return None
        return value

    @staticmethod
    def _format_metric_summary(metrics: Dict[str, float]) -> str:
        parts = []
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, float) and math.isfinite(value):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def _stdout_epoch_summary(self) -> None:
        callback_metrics = getattr(self.trainer, "callback_metrics", {})
        train_summary = {}
        val_summary = {}
        # Lightning accumulates both step and epoch outputs in callback_metrics; filter here so the
        # stdout summary only prints scalar loss/validation metrics that are meaningful to monitor.
        for raw_key, raw_value in callback_metrics.items():
            key = str(raw_key)
            value = self._scalar_metric_value(raw_value)
            if value is None:
                continue
            if key.startswith("Val/"):
                val_summary[key] = value
            elif key.startswith("val_"):
                val_summary[key] = value
            elif key.startswith("Train/") and "loss" in key.lower():
                train_summary[key] = value

        if not train_summary and self._last_train_total_loss is not None:
            train_summary["Train/total_loss_last"] = self._last_train_total_loss

        epoch_num = self.current_epoch + 1
        if train_summary:
            print(
                f"TRAIN_JVP_TRAIN_SUMMARY epoch={epoch_num} {self._format_metric_summary(train_summary)}",
                flush=True,
            )
        if val_summary:
            print(
                f"TRAIN_JVP_VAL_SUMMARY epoch={epoch_num} {self._format_metric_summary(val_summary)}",
                flush=True,
            )

@hydra.main(version_base=None, config_path="configs")
def main(cfg=None):

    # Load TVM YAML config
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))
    args = argparse.Namespace(**OmegaConf.to_container(cfg.pl, resolve=True))
    ddp_process_group_backend = getattr(args, "ddp_process_group_backend", None)
    if hasattr(args, "ddp_process_group_backend"):
        del args.ddp_process_group_backend
    ddp_find_unused_parameters = getattr(args, "ddp_find_unused_parameters", None)
    if hasattr(args, "ddp_find_unused_parameters"):
        del args.ddp_find_unused_parameters
    if str(args.strategy).startswith("ddp") and (
        ddp_process_group_backend or ddp_find_unused_parameters is not None
    ):
        strategy_kwargs = {}
        if ddp_process_group_backend:
            strategy_kwargs["process_group_backend"] = str(ddp_process_group_backend)
        if ddp_find_unused_parameters is not None:
            strategy_kwargs["find_unused_parameters"] = bool(ddp_find_unused_parameters)
        args.strategy = DDPStrategy(**strategy_kwargs)

    # Make Deterministic / Benchmark settings
    setup_cudnn_and_tf32(cfg)
    pl.seed_everything(cfg.training.seed, workers=True)

    # Select batch size per GPU
    batch_size = cfg.training.batch_size
    batch_gpu = cfg.training.batch_gpu

    num_devices = args.devices if isinstance(args.devices, int) else len(args.devices)
    batch_gpu_total = batch_size // num_devices
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total

    val_batch_gpu = cfg.training.val_batch_gpu if cfg.training.val_batch_gpu > 0 else batch_gpu

    # Logging / checkpoint config
    logging_cfg = getattr(cfg, "logging", OmegaConf.create({}))

    # Datasets
    return_uint = True
    train_dataloader = build_train_loader(cfg, batch_gpu=batch_gpu, return_uint=return_uint)
    if isinstance(getattr(train_dataloader, "batch_sampler", None), RandomBucketBatchSampler):
        args.use_distributed_sampler = False
        _disable_encoder_compile_for_random_buckets(cfg)
    val_input_res = cfg.ir.val_input_res
    val_sets = cfg.dataset.get("val_sets", None)
    if val_sets is None:
        val_dset = build_val_dataset(
            cfg.dataset.get("val_set", None),
            default_path=cfg.dataset.path,
            patch_size=val_input_res,
            return_uint=return_uint,
        )
        val_dataloader = DataLoader(
            val_dset,
            batch_size=val_batch_gpu,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        val_entries = [_plain_cfg(entry) for entry in val_sets]
        val_names = [str(entry.get("name", f"val_{idx}")) for idx, entry in enumerate(val_entries)]
        cfg.dataset.val_names = val_names
        val_dataloader = [
            DataLoader(
                build_val_dataset(entry, default_path=cfg.dataset.path, patch_size=val_input_res, return_uint=return_uint),
                batch_size=val_batch_gpu,
                shuffle=False,
                num_workers=cfg.dataloader.num_workers,
            )
            for entry in val_entries
        ]
        print(f"[dataset] multi-val loaders: {', '.join(val_names)}", flush=True)

    # Model
    model = TrainJVPLightningModule(cfg)


    # Callbacks: checkpoints + LR logging
    ckpt_dir = str(getattr(logging_cfg, "checkpoint_dir", "outputs/pl_runs"))
    max_epochs = getattr(cfg.pl, "max_epochs", 10)
    args.max_epochs = max_epochs
    args.gradient_clip_val = cfg.training.get("max_grad_norm", 0.0) or 0.0
    args.gradient_clip_algorithm = "norm"
    resume_ckpt = getattr(args, "resume_ckpt", None)
    if hasattr(args, "resume_ckpt"):
        del args.resume_ckpt

    # Checkpoints
    checkpoint_every_n_epochs = int(logging_cfg.get("checkpoint_every_n_epochs", logging_cfg.get("save_epoch_every_n_epochs", 1)))
    ckpt_every_epoch = ModelCheckpoint(
        dirpath=ckpt_dir,  # local temp dir (Hydra will change cwd)
        filename="epoch{epoch:04d}",
        save_top_k=-1,  # keep all epochs
        every_n_epochs=checkpoint_every_n_epochs,
        save_last=True,
        enable_version_counter=False,
    )
    best_monitor = str(getattr(logging_cfg, "best_monitor", "Val/total_loss"))
    best_mode = str(getattr(logging_cfg, "best_mode", "min"))
    best_filename = str(getattr(logging_cfg, "best_filename", "best-epoch{epoch:04d}-valloss{Val/total_loss:.4f}"))
    ckpt_best_val = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=best_filename,
        monitor=best_monitor,
        mode=best_mode,
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
    )
    ckpt_best_callbacks = [ckpt_best_val]
    best_monitors_cfg = logging_cfg.get("best_monitors", None)
    if best_monitors_cfg is not None:
        ckpt_best_callbacks = []
        for entry in best_monitors_cfg:
            entry = _plain_cfg(entry)
            ckpt_best_callbacks.append(
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename=str(entry.get("filename", "best-metric")),
                    monitor=str(entry["monitor"]),
                    mode=str(entry.get("mode", "max")),
                    save_top_k=int(entry.get("save_top_k", 1)),
                    save_last=False,
                    enable_version_counter=False,
                    auto_insert_metric_name=False,
                )
            )
    callbacks = [ckpt_every_epoch, *ckpt_best_callbacks]

    print("Logging mode: stdout/local")
    print(f"Checkpoint dir: {ckpt_dir}")

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=False,
        **vars(args),
    )

    print(f"Lightning resume ckpt: {resume_ckpt or ''}", flush=True)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_ckpt if resume_ckpt else None,
        weights_only=False if resume_ckpt else None,
    )


if __name__ == "__main__":
    main()
