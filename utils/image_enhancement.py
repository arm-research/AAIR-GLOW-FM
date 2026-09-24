# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Utilities for loading and running the Glow-FM image enhancement model."""

import copy
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.generate_images import generator_fn
from utils import resolver

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve_device(value: str) -> torch.device:
    """Resolve a requested Torch device and fail if it is unavailable."""
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    mps_backend = getattr(torch.backends, "mps", None)
    if device.type == "mps" and (mps_backend is None or not mps_backend.is_available()):
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false")
    return device


def _model_state(checkpoint: dict) -> dict:
    """Extract the model state dictionary from a checkpoint."""
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a state dictionary")

    prefix = "model."
    selected = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError("Checkpoint does not contain 'model' weights")
    return selected


def load_pipeline(checkpoint_path: Path, device: torch.device):
    """Construct TAESD and HDiT from a checkpoint and load weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("hyper_parameters")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a hyper_parameters configuration")

    # Construct the encoder and network from the checkpoint configuration.
    encoder_config = copy.deepcopy(config["encoder"])
    image_resolution = int(encoder_config.pop("res", 32))
    encoder_config.pop("dim_latent", None)
    encoder_config["use_compile"] = False
    encoder = resolver.instantiate(**encoder_config)
    latent_channels = int(encoder.latent_channels)

    network_config = copy.deepcopy(config["network"])
    network_config["compile"] = False
    if device.type != "cuda":
        network_config["data_type"] = "fp32"
    network = resolver.instantiate(
        **network_config,
        img_resolution=image_resolution,
        img_channels=latent_channels,
    )
    network.load_state_dict(_model_state(checkpoint), strict=True)
    network.eval().requires_grad_(False).to(device)
    encoder.init(device)
    return network, encoder


def pad_to_multiple(x: torch.Tensor, multiple: int):
    """Pad NCHW spatial dimensions on the bottom and right."""
    height, width = x.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height == 0 and pad_width == 0:
        return x, (height, width)
    mode = "reflect" if pad_height < height and pad_width < width else "replicate"
    return F.pad(x, (0, pad_width, 0, pad_height), mode=mode), (height, width)


def unpad(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Restore the original height and width after bottom/right padding."""
    return x[..., : shape[0], : shape[1]]


def read_image(path: Path, device: torch.device) -> torch.Tensor:
    """Load one image as a uint8 RGB tensor with shape [1, 3, H, W]."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def write_image(image: torch.Tensor, path: Path) -> None:
    """Save a uint8 RGB tensor with shape [1, 3, H, W]."""
    image = image.detach().squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def _is_archive_metadata(path: Path) -> bool:
    """Identify macOS metadata files sometimes included in dataset archives."""
    return path.name.startswith("._") or "__MACOSX" in path.parts


def _is_under(path: Path, root: Path) -> bool:
    """Return whether path is root itself or one of its descendants."""
    return path == root or root in path.parents


def input_images(path: Path, *, exclude: Path | None = None) -> Iterable[Path]:
    """Find supported images deterministically, excluding outputs and metadata."""
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES or _is_archive_metadata(path):
            raise ValueError(f"Unsupported input image type: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    excluded_root = exclude.resolve() if exclude is not None else None
    images = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and not _is_archive_metadata(item)
        and (excluded_root is None or not _is_under(item.resolve(), excluded_root))
    )
    if not images:
        raise ValueError(f"No supported images found under {path}")
    return images


def index_images(path: Path, *, exclude: Path | None = None) -> tuple[Path, dict[str, Path]]:
    """Index images by their extension-free path relative to the input root.

    Evaluation uses these keys to pair low-light and ground-truth images even
    when corresponding files use different image extensions.
    """
    root = path if path.is_dir() else path.parent
    index: dict[str, Path] = {}
    for image_path in input_images(path, exclude=exclude):
        key = image_path.relative_to(root).with_suffix("").as_posix()
        if key in index:
            raise ValueError(f"Duplicate image key {key!r}: {index[key]} and {image_path}")
        index[key] = image_path
    return root, index


def prediction_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    """Preserve the relative input path and append ``_pred`` to its stem."""
    relative = input_path.relative_to(input_root)
    return output_root / relative.parent / f"{relative.stem}_pred.png"


@torch.inference_mode()
def enhance(
    network,
    encoder,
    image: torch.Tensor,
    *,
    nfe: int,
    sigma: float,
) -> torch.Tensor:
    """Enhance a low-light image using the Glow-FM model."""
    # TAESD downsamples by eight. Latent padding to 32.
    image, image_shape = pad_to_multiple(image, 8)
    latent = encoder.encode(image)
    latent, latent_shape = pad_to_multiple(latent, 32)

    prior = latent
    if sigma:
        generator = torch.Generator(device=latent.device).manual_seed(1234)
        prior = latent + torch.randn(
            latent.shape,
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        ) * sigma

    prediction = generator_fn(
        network,
        prior,
        class_labels=None,
        num_steps=nfe,
        cfg_scale=None,
        use_time_gap=False,
    )
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Model produced a non-finite latent prediction")

    # Remove latent padding before decoding, then restore the input image size.
    prediction = unpad(prediction, latent_shape)
    prediction = encoder.decode(prediction)
    return unpad(prediction, image_shape)
