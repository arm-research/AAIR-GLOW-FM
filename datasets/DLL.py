# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import os
import json
import io
from typing import Optional, Callable, Tuple, Dict, Any, List, Union
import multiprocessing as mp
from typing import Optional, Callable, Tuple, Dict, Any, List

import numpy as np
import torch
from numba.tests.test_pycc import base_path
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import math
import torch
from torchvision.transforms.v2 import functional as F
from PIL import Image

class PairedPadToSize:
    """
    Center-pad paired images to at least target_size before crop.

    Reflection padding avoids introducing black borders for real images that are
    slightly smaller than the requested training crop. Very tiny dimensions fall
    back to edge padding because reflection is undefined for 1-pixel axes.
    """

    def __init__(self, target_size: Union[int, Tuple[int, int]], mode: str = "reflect", fill: int = 0):
        if isinstance(target_size, int):
            target_h = target_w = int(target_size)
        else:
            target_h, target_w = int(target_size[0]), int(target_size[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        mode = str(mode).lower()
        if mode == "replicate":
            mode = "edge"
        if mode not in {"reflect", "edge", "symmetric", "constant"}:
            raise ValueError(f"Unsupported crop padding mode: {mode}")
        self.target_h = target_h
        self.target_w = target_w
        self.mode = mode
        self.fill = int(fill)

    @staticmethod
    def _get_hw(img) -> Tuple[int, int]:
        if isinstance(img, Image.Image):
            w, h = img.size
            return h, w
        return int(img.shape[-2]), int(img.shape[-1])

    def _padding(self, img) -> Tuple[int, int, int, int]:
        h, w = self._get_hw(img)
        pad_h = max(0, self.target_h - h)
        pad_w = max(0, self.target_w - w)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return left, top, right, bottom

    def _numpy_mode_for(self, h: int, w: int, top: int, bottom: int, left: int, right: int) -> str:
        if self.mode != "reflect":
            return self.mode
        if (top or bottom) and h <= 1:
            return "edge"
        if (left or right) and w <= 1:
            return "edge"
        return "reflect"

    def _pad_pil(self, img: Image.Image, padding: Tuple[int, int, int, int]) -> Image.Image:
        left, top, right, bottom = padding
        if left == top == right == bottom == 0:
            return img

        arr = np.asarray(img)
        h, w = arr.shape[:2]
        pad_spec = ((top, bottom), (left, right)) if arr.ndim == 2 else ((top, bottom), (left, right), (0, 0))
        mode = self._numpy_mode_for(h, w, top, bottom, left, right)
        if mode == "constant":
            padded = np.pad(arr, pad_spec, mode=mode, constant_values=self.fill)
        else:
            padded = np.pad(arr, pad_spec, mode=mode)
        return Image.fromarray(padded)

    def _pad_tensor(self, img: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
        left, top, right, bottom = padding
        if left == top == right == bottom == 0:
            return img

        h, w = int(img.shape[-2]), int(img.shape[-1])
        mode = self.mode
        if mode == "edge":
            mode = "replicate"
        if mode == "reflect" and (((top or bottom) and max(top, bottom) >= h) or ((left or right) and max(left, right) >= w)):
            mode = "replicate"
        if mode == "constant":
            return torch.nn.functional.pad(img, (left, right, top, bottom), mode=mode, value=float(self.fill))
        return torch.nn.functional.pad(img, (left, right, top, bottom), mode=mode)

    def _pad_one(self, img):
        padding = self._padding(img)
        if isinstance(img, Image.Image):
            return self._pad_pil(img, padding)
        if torch.is_tensor(img):
            return self._pad_tensor(img, padding)
        raise TypeError(f"Unsupported image type for padding: {type(img)!r}")

    def __call__(self, *imgs):
        padded = tuple(self._pad_one(img) for img in imgs)
        return padded[0] if len(padded) == 1 else padded

class PairedNearCenterRandomCrop:
    """
    Sample crop center from Normal(image_center, sigma), clamp to valid range,
    then crop patch_size x patch_size. Same params for low & gt.

    sigma_frac controls how "tight" the crops are around center:
      - smaller -> more concentrated near center
      - larger  -> more spread toward edges
    """
    def __init__(self, patch_size: int, sigma_frac: float = 0.15):
        self.patch_size = int(patch_size)
        self.sigma_frac = float(sigma_frac)

    def _get_hw(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
            return h, w
        # torch tensor: [..., H, W]
        return int(img.shape[-2]), int(img.shape[-1])

    def __call__(self, low, gt):
        ps = self.patch_size
        if ps == -1:
            return low, gt

        h, w = self._get_hw(low)

        if h < ps or w < ps:
            raise ValueError(f"Image too small for crop: got {h}x{w}, patch_size={ps}")

        # valid top-left ranges
        max_top = h - ps
        max_left = w - ps

        # sample crop center around image center
        cx0 = w / 2.0
        cy0 = h / 2.0

        # sigma relative to image size (you can tune this)
        sigma_x = self.sigma_frac * w
        sigma_y = self.sigma_frac * h

        cx = torch.normal(mean=torch.tensor(cx0), std=torch.tensor(sigma_x)).item()
        cy = torch.normal(mean=torch.tensor(cy0), std=torch.tensor(sigma_y)).item()

        # convert to top-left, then clamp so crop fits in image
        left = int(round(cx - ps / 2.0))
        top  = int(round(cy - ps / 2.0))

        left = max(0, min(left, max_left))
        top  = max(0, min(top,  max_top))

        low_c = F.crop(low, top, left, ps, ps)
        gt_c  = F.crop(gt,  top, left, ps, ps)
        return low_c, gt_c

def paired_rot90(low, gt, p=1.0):
    if p <= 0 or torch.rand(()) >= p:
        return low, gt
    k = int(torch.randint(0, 2, (1,)).item())  # 0,1,2,3
    if k == 0:
        return low, gt
    return torch.rot90(low, k, dims=(-2, -1)), torch.rot90(gt, k, dims=(-2, -1))


def _cfg_dict(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "items"):
        return dict(cfg.items())
    return {}


def _prob(cfg: Dict[str, Any], key: str, default: float) -> float:
    return max(0.0, min(1.0, float(cfg.get(key, default))))


def _range_pair(value, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if len(value) != 2:
        raise ValueError(f"Expected a two-value range, got {value}")
    lo, hi = float(value[0]), float(value[1])
    if hi < lo:
        raise ValueError(f"Invalid range with high < low: {value}")
    return lo, hi


def _choice_int(value, default: List[int]) -> int:
    choices = list(value) if value is not None else list(default)
    if not choices:
        raise ValueError("choices must not be empty")
    idx = int(torch.randint(0, len(choices), (1,)).item())
    return int(choices[idx])


def _uniform(lo: float, hi: float) -> float:
    return float(torch.empty(()).uniform_(float(lo), float(hi)).item())


def _interpolation_mode(name: str):
    key = str(name).lower()
    mapping = {
        "nearest": v2.InterpolationMode.NEAREST,
        "bilinear": v2.InterpolationMode.BILINEAR,
        "bicubic": v2.InterpolationMode.BICUBIC,
        "lanczos": v2.InterpolationMode.LANCZOS,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported interpolation mode: {name}")
    return mapping[key]


def _is_aug_config_present(config: Optional[Dict[str, Any]]) -> bool:
    return bool(_cfg_dict(config))


def _aug_enabled(augment_config: Optional[Dict[str, Any]], legacy_use_aug: bool) -> bool:
    if not _is_aug_config_present(augment_config):
        return bool(legacy_use_aug)
    return bool(augment_config.get("enabled", False))


def _resize_aug_enabled(augment_config: Optional[Dict[str, Any]]) -> bool:
    if not _is_aug_config_present(augment_config):
        return False
    resize_cfg = _cfg_dict(augment_config.get("resize_aug"))
    return bool(resize_cfg.get("enabled", False))


def _legacy_geometry_config() -> Dict[str, Any]:
    return {"hflip_p": 0.5, "vflip_p": 0.0, "rot90_p": 0.5, "rot90_choices": [0, 1]}


def _geometry_config(augment_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not _is_aug_config_present(augment_config):
        return _legacy_geometry_config()
    cfg = _cfg_dict(augment_config.get("paired_geometry"))
    return {
        "hflip_p": _prob(cfg, "hflip_p", 0.5),
        "vflip_p": _prob(cfg, "vflip_p", 0.0),
        "rot90_p": _prob(cfg, "rot90_p", 0.0),
        "rot90_choices": list(cfg.get("rot90_choices", [1, 2, 3])),
    }


def _resize_aug_config(augment_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = _cfg_dict(_cfg_dict(augment_config).get("resize_aug"))
    return {
        "mode": str(cfg.get("mode", "short_side_then_crop")),
        "short_side_choices": cfg.get("short_side_choices", [512, 768, 1024]),
        "scale_range": cfg.get("scale_range", None),
        "interpolation": str(cfg.get("interpolation", "bicubic")),
        "antialias": bool(cfg.get("antialias", True)),
    }


def _lq_only_config(augment_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = _cfg_dict(_cfg_dict(augment_config).get("lq_only"))
    return cfg if bool(cfg.get("enabled", False)) else {"enabled": False}


def _get_hw(img) -> Tuple[int, int]:
    if isinstance(img, Image.Image):
        w, h = img.size
        return h, w
    return int(img.shape[-2]), int(img.shape[-1])


def _paired_resize_aug(low_img, gt_img, cfg: Dict[str, Any]):
    mode = cfg["mode"]
    if mode not in {"short_side", "short_side_then_crop"}:
        raise ValueError(f"Unsupported resize_aug.mode={mode!r}; expected short_side_then_crop")

    h, w = _get_hw(low_img)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size for resize augmentation: {h}x{w}")

    short_side_choices = cfg.get("short_side_choices")
    scale_range = cfg.get("scale_range")
    if short_side_choices:
        target_short = _choice_int(short_side_choices, [min(h, w)])
    else:
        target_short = min(h, w)
    if scale_range is not None:
        lo, hi = _range_pair(scale_range, (1.0, 1.0))
        target_short = max(1, int(round(float(target_short) * _uniform(lo, hi))))

    scale = float(target_short) / float(min(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    interpolation = _interpolation_mode(cfg.get("interpolation", "bicubic"))
    antialias = bool(cfg.get("antialias", True))
    low_img = F.resize(low_img, [new_h, new_w], interpolation=interpolation, antialias=antialias)
    gt_img = F.resize(gt_img, [new_h, new_w], interpolation=interpolation, antialias=antialias)
    return low_img, gt_img


def _size_pair(value: Union[int, Tuple[int, int], List[int]], *, name: str = "size") -> Tuple[int, int]:
    if isinstance(value, int):
        h = w = int(value)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or [h, w], got {value}")
        h, w = int(value[0]), int(value[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must contain positive dimensions, got {value}")
    return h, w


class PairedRandomCrop:
    def __init__(self, patch_size: Union[int, Tuple[int, int], List[int]]):
        self.crop_h, self.crop_w = _size_pair(patch_size, name="patch_size")

    def __call__(self, low_img, gt_img):
        h, w = _get_hw(low_img)
        if h < self.crop_h or w < self.crop_w:
            raise ValueError(f"Image too small for crop: got {h}x{w}, crop_size={self.crop_h}x{self.crop_w}")
        top = int(torch.randint(0, h - self.crop_h + 1, (1,)).item())
        left = int(torch.randint(0, w - self.crop_w + 1, (1,)).item())
        return (
            F.crop(low_img, top, left, self.crop_h, self.crop_w),
            F.crop(gt_img, top, left, self.crop_h, self.crop_w),
        )


def _fixed_crop_size(patch_size: int, crop_h: Optional[int], crop_w: Optional[int]) -> Optional[Tuple[int, int]]:
    if (crop_h is None) ^ (crop_w is None):
        raise ValueError("crop_h and crop_w must be configured together")
    if crop_h is not None and crop_w is not None:
        crop_h, crop_w = int(crop_h), int(crop_w)
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(f"crop_h and crop_w must be positive, got crop_h={crop_h} crop_w={crop_w}")
        return crop_h, crop_w
    patch_size = int(patch_size)
    if patch_size == -1:
        return None
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive or -1, got {patch_size}")
    return patch_size, patch_size


def _random_bucket_crop_cfg(
    augment_config: Optional[Dict[str, Any]],
    explicit_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    augment_bucket = _cfg_dict(_cfg_dict(augment_config).get("random_bucket_crop"))
    explicit_bucket = _cfg_dict(explicit_config)
    if not augment_bucket and not explicit_bucket:
        return {"enabled": False}
    return {**augment_bucket, **explicit_bucket}


class RandomBucketCropPolicy:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = _cfg_dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        if not self.enabled:
            self.min_h = self.min_w = self.max_h = self.max_w = 0
            self.align = 1
            self.fixed_buckets: List[Tuple[int, int]] = []
            self.fixed_bucket_prob = 0.0
            self.random_bucket_count = 0
            self.random_h_values: List[int] = []
            self.random_w_values: List[int] = []
            return

        mode = str(self.config.get("mode", "crop_pad")).lower()
        if mode != "crop_pad":
            raise ValueError(f"Unsupported random_bucket_crop.mode={mode!r}; expected 'crop_pad'")

        self.min_h = int(self.config.get("min_h", 0))
        self.min_w = int(self.config.get("min_w", 0))
        self.max_h = int(self.config.get("max_h", 0))
        self.max_w = int(self.config.get("max_w", 0))
        self.align = int(self.config.get("align", 1))
        if self.align <= 0:
            raise ValueError(f"random_bucket_crop.align must be positive, got {self.align}")
        for key, value in (("min_h", self.min_h), ("min_w", self.min_w), ("max_h", self.max_h), ("max_w", self.max_w)):
            if value <= 0:
                raise ValueError(f"random_bucket_crop.{key} must be positive, got {value}")
        if self.min_h > self.max_h:
            raise ValueError("random_bucket_crop.min_h must be <= max_h")
        if self.min_w > self.max_w:
            raise ValueError("random_bucket_crop.min_w must be <= max_w")

        self.fixed_buckets = [
            _size_pair(bucket, name="random_bucket_crop.fixed_buckets item")
            for bucket in list(self.config.get("fixed_buckets") or [])
        ]
        self.fixed_bucket_prob = float(self.config.get("fixed_bucket_prob", 0.0))
        if not 0.0 <= self.fixed_bucket_prob <= 1.0:
            raise ValueError("random_bucket_crop.fixed_bucket_prob must be in [0, 1]")
        if self.fixed_bucket_prob > 0.0 and not self.fixed_buckets:
            raise ValueError("random_bucket_crop.fixed_buckets must be non-empty when fixed_bucket_prob > 0")

        self.random_bucket_count = int(self.config.get("random_bucket_count", 0) or 0)
        if self.random_bucket_count < 0:
            raise ValueError("random_bucket_crop.random_bucket_count must be non-negative")
        self.random_h_values = self._aligned_values(self.min_h, self.max_h, "height")
        self.random_w_values = self._aligned_values(self.min_w, self.max_w, "width")
        self.random_buckets = self._random_bucket_pool()
        if self.fixed_bucket_prob < 1.0 and not self.random_buckets:
            raise ValueError("random_bucket_crop has no valid aligned random bucket dimensions")

    def _aligned_values(self, lo: int, hi: int, label: str) -> List[int]:
        first = int(math.ceil(float(lo) / float(self.align)) * self.align)
        if first > hi:
            raise ValueError(
                f"random_bucket_crop has no {label} multiple of align={self.align} in [{lo}, {hi}]"
            )
        return list(range(first, hi + 1, self.align))

    def _random_bucket_pool(self) -> List[Tuple[int, int]]:
        buckets = [(h, w) for h in self.random_h_values for w in self.random_w_values]
        if self.random_bucket_count <= 0 or self.random_bucket_count >= len(buckets):
            return buckets
        if self.random_bucket_count == 1:
            return [buckets[len(buckets) // 2]]
        indices = np.linspace(0, len(buckets) - 1, num=self.random_bucket_count).round().astype(int)
        return [buckets[int(idx)] for idx in sorted(set(indices.tolist()))]

    def sample_shape(self) -> Tuple[int, int]:
        if not self.enabled:
            raise RuntimeError("Cannot sample from a disabled random bucket crop policy")
        use_fixed = bool(self.fixed_buckets) and torch.rand(()) < self.fixed_bucket_prob
        if use_fixed:
            idx = int(torch.randint(0, len(self.fixed_buckets), (1,)).item())
            return self.fixed_buckets[idx]
        idx = int(torch.randint(0, len(self.random_buckets), (1,)).item())
        return self.random_buckets[idx]

    def describe(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "mode": "crop_pad",
            "min_h": self.min_h,
            "min_w": self.min_w,
            "max_h": self.max_h,
            "max_w": self.max_w,
            "align": self.align,
            "fixed_buckets": self.fixed_buckets,
            "fixed_bucket_prob": self.fixed_bucket_prob,
            "random_bucket_count": self.random_bucket_count,
            "random_bucket_pool_size": len(self.random_buckets),
        }


def _unpack_bucket_index(index) -> Tuple[int, Optional[Tuple[int, int]]]:
    if isinstance(index, tuple) and len(index) == 3:
        sample_idx, crop_h, crop_w = index
        return int(sample_idx), (int(crop_h), int(crop_w))
    return int(index), None


class RandomBucketBatchSampler:
    """Wrap a sample sampler and attach one random bucket crop shape to each batch."""

    def __init__(
        self,
        sampler,
        batch_size: int,
        policy: RandomBucketCropPolicy,
        drop_last: bool,
        seed: int = 0,
    ):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.policy = policy
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not getattr(policy, "enabled", False):
            raise ValueError("RandomBucketBatchSampler requires an enabled RandomBucketCropPolicy")

    def _sample_shape_for_batch(self, batch_idx: int) -> Tuple[int, int]:
        # In DDP, every rank must use the same crop size for the same batch
        # position, otherwise fast ranks can reach collectives while slower
        # ranks are still processing a larger shape.
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003 + int(batch_idx))
        use_fixed = (
            bool(self.policy.fixed_buckets)
            and torch.rand((), generator=generator) < self.policy.fixed_bucket_prob
        )
        if use_fixed:
            idx = int(torch.randint(0, len(self.policy.fixed_buckets), (1,), generator=generator).item())
            return self.policy.fixed_buckets[idx]
        idx = int(torch.randint(0, len(self.policy.random_buckets), (1,), generator=generator).item())
        return self.policy.random_buckets[idx]

    def __iter__(self):
        batch = []
        batch_idx = 0
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) == self.batch_size:
                crop_h, crop_w = self._sample_shape_for_batch(batch_idx)
                yield [(sample_idx, crop_h, crop_w) for sample_idx in batch]
                batch = []
                batch_idx += 1
        if batch and not self.drop_last:
            crop_h, crop_w = self._sample_shape_for_batch(batch_idx)
            yield [(sample_idx, crop_h, crop_w) for sample_idx in batch]

    def __len__(self):
        sampler_len = len(self.sampler)
        if self.drop_last:
            return sampler_len // self.batch_size
        return int(math.ceil(float(sampler_len) / float(self.batch_size)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class BucketAwareConcatDataset(Dataset):
    def __init__(self, datasets):
        self.dataset = torch.utils.data.ConcatDataset(datasets)

    @property
    def datasets(self):
        return self.dataset.datasets

    @property
    def cumulative_sizes(self):
        return self.dataset.cumulative_sizes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample_idx, bucket_shape = _unpack_bucket_index(index)
        if bucket_shape is None:
            return self.dataset[sample_idx]
        if sample_idx < 0:
            if -sample_idx > len(self.dataset):
                raise ValueError("absolute value of index should not exceed dataset length")
            sample_idx = len(self.dataset) + sample_idx
        dataset_idx = int(np.searchsorted(self.dataset.cumulative_sizes, sample_idx, side="right"))
        if dataset_idx == 0:
            child_idx = sample_idx
        else:
            child_idx = sample_idx - self.dataset.cumulative_sizes[dataset_idx - 1]
        crop_h, crop_w = bucket_shape
        return self.dataset.datasets[dataset_idx][(child_idx, crop_h, crop_w)]


class BucketAwareSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_idx, bucket_shape = _unpack_bucket_index(index)
        mapped_idx = int(self.indices[sample_idx])
        if bucket_shape is None:
            return self.dataset[mapped_idx]
        crop_h, crop_w = bucket_shape
        return self.dataset[(mapped_idx, crop_h, crop_w)]


def _dataset_random_bucket_policy(dataset):
    if hasattr(dataset, "random_bucket_crop_policy"):
        policy = dataset.random_bucket_crop_policy
        return policy if getattr(policy, "enabled", False) else None
    return None


def _random_bucket_policy_signature(policy: RandomBucketCropPolicy) -> Tuple[Any, ...]:
    return (
        policy.min_h,
        policy.min_w,
        policy.max_h,
        policy.max_w,
        policy.align,
        tuple(policy.fixed_buckets),
        float(policy.fixed_bucket_prob),
        int(policy.random_bucket_count),
        tuple(policy.random_buckets),
    )


def random_bucket_policy_from_dataset(dataset):
    direct_policy = _dataset_random_bucket_policy(dataset)
    if direct_policy is not None:
        return direct_policy
    if hasattr(dataset, "dataset"):
        return random_bucket_policy_from_dataset(dataset.dataset)
    if hasattr(dataset, "datasets"):
        children = list(dataset.datasets)
        policies = [_dataset_random_bucket_policy(child) for child in children]
        enabled = [policy for policy in policies if policy is not None]
        if not enabled:
            return None
        if len(enabled) != len(children):
            raise ValueError(
                "random_bucket_crop must be enabled with the same policy for every train dataset in a multi-dataset loader"
            )
        first = enabled[0]
        first_sig = _random_bucket_policy_signature(first)
        if any(_random_bucket_policy_signature(policy) != first_sig for policy in enabled[1:]):
            raise ValueError(
                "random_bucket_crop policies must match across all train datasets in a multi-dataset loader"
            )
        return first
    return None


class PairedGeometryAugment:
    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        rot90_p: float = 0.0,
        rot90_choices: Optional[List[int]] = None,
    ):
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.rot90_p = float(rot90_p)
        self.rot90_choices = [int(k) % 4 for k in (rot90_choices or [1, 2, 3])]

    def __call__(self, low_img, gt_img):
        if self.hflip_p > 0 and torch.rand(()) < self.hflip_p:
            low_img, gt_img = F.hflip(low_img), F.hflip(gt_img)
        if self.vflip_p > 0 and torch.rand(()) < self.vflip_p:
            low_img, gt_img = F.vflip(low_img), F.vflip(gt_img)
        if self.rot90_p > 0 and torch.rand(()) < self.rot90_p:
            k = _choice_int(self.rot90_choices, [1, 2, 3])
            h, w = _get_hw(low_img)
            if h != w and k % 2:
                k = 0
            if k:
                low_img, gt_img = torch.rot90(low_img, k, dims=(-2, -1)), torch.rot90(gt_img, k, dims=(-2, -1))
        return low_img, gt_img


class LQOnlyAugment:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = _cfg_dict(config)
        self.enabled = bool(self.config.get("enabled", False))

    @staticmethod
    def _to_float01(img: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype, bool]:
        dtype = img.dtype
        is_uint8 = dtype == torch.uint8
        x = img.float()
        if is_uint8:
            x = x / 255.0
        return x.clamp(0.0, 1.0), dtype, is_uint8

    @staticmethod
    def _restore_dtype(img: torch.Tensor, dtype: torch.dtype, is_uint8: bool) -> torch.Tensor:
        img = img.clamp(0.0, 1.0)
        if is_uint8:
            return (img * 255.0).round().clamp(0, 255).to(torch.uint8)
        return img.to(dtype)

    @staticmethod
    def _maybe(p: float) -> bool:
        return p > 0 and torch.rand(()) < p

    def _blur(self, img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        if kernel_size <= 1:
            return img
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - (kernel_size - 1) / 2.0
        kernel1d = torch.exp(-(coords ** 2) / (2.0 * max(float(sigma), 1e-6) ** 2))
        kernel1d = kernel1d / kernel1d.sum()
        kernel2d = (kernel1d[:, None] * kernel1d[None, :]).expand(img.shape[0], 1, kernel_size, kernel_size)
        padded = img.unsqueeze(0)
        pad = kernel_size // 2
        pad_mode = "reflect" if pad < min(int(img.shape[-2]), int(img.shape[-1])) else "replicate"
        padded = torch.nn.functional.pad(padded, (pad,) * 4, mode=pad_mode)
        return torch.nn.functional.conv2d(padded, kernel2d, groups=img.shape[0]).squeeze(0)

    def _jpeg(self, img: torch.Tensor, quality: int) -> torch.Tensor:
        arr = (img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        with io.BytesIO() as buf:
            Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
            buf.seek(0)
            out = np.asarray(Image.open(buf).convert("RGB"))
        return torch.from_numpy(out).permute(2, 0, 1).to(device=img.device, dtype=img.dtype) / 255.0

    def _down_up(self, img: torch.Tensor, scale: float, interpolation: str) -> torch.Tensor:
        _, h, w = img.shape
        down_h = max(1, int(round(h * scale)))
        down_w = max(1, int(round(w * scale)))
        mode = "bicubic" if str(interpolation).lower() == "bicubic" else "bilinear"
        x = img.unsqueeze(0)
        x = torch.nn.functional.interpolate(x, size=(down_h, down_w), mode=mode, align_corners=False)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode=mode, align_corners=False)
        return x.squeeze(0)

    def __call__(self, low_img: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return low_img

        cfg = self.config
        x, dtype, is_uint8 = self._to_float01(low_img)

        if self._maybe(_prob(cfg, "gamma_p", 0.0)):
            lo, hi = _range_pair(cfg.get("gamma_range"), (0.7, 1.5))
            x = x.clamp(0.0, 1.0).pow(_uniform(lo, hi))

        if self._maybe(_prob(cfg, "exposure_p", 0.0)):
            lo, hi = _range_pair(cfg.get("exposure_gain_range"), (0.7, 1.4))
            x = x * _uniform(lo, hi)

        if self._maybe(_prob(cfg, "channel_gain_p", 0.0)):
            lo, hi = _range_pair(cfg.get("channel_gain_range"), (0.9, 1.1))
            gains = torch.empty((x.shape[0], 1, 1), device=x.device, dtype=x.dtype).uniform_(lo, hi)
            x = x * gains

        if self._maybe(_prob(cfg, "gaussian_noise_p", 0.0)):
            lo, hi = _range_pair(cfg.get("gaussian_sigma_range"), (0.0, 0.02))
            sigma = _uniform(lo, hi)
            x = x + torch.randn_like(x) * sigma

        x = x.clamp(0.0, 1.0)

        if self._maybe(_prob(cfg, "blur_p", 0.0)):
            kernel_size = _choice_int(cfg.get("blur_kernel_choices"), [3, 5])
            lo, hi = _range_pair(cfg.get("blur_sigma_range"), (0.1, 1.2))
            x = self._blur(x, kernel_size=kernel_size, sigma=_uniform(lo, hi))

        if self._maybe(_prob(cfg, "jpeg_p", 0.0)):
            lo, hi = _range_pair(cfg.get("jpeg_quality_range"), (60.0, 95.0))
            x = self._jpeg(x, quality=int(round(_uniform(lo, hi))))

        if self._maybe(_prob(cfg, "down_up_p", 0.0)):
            lo, hi = _range_pair(cfg.get("downscale_range"), (0.5, 0.9))
            x = self._down_up(x, scale=_uniform(lo, hi), interpolation=cfg.get("interpolation", "bicubic"))

        return self._restore_dtype(x, dtype, is_uint8)


def _paired_low_stem_key(stem: str) -> str:
    """Normalize low-side paired stems such as r00816405t or low00001."""
    if len(stem) > 1 and stem[0].lower() == "r" and stem[1].isdigit():
        return stem[1:]
    lower = stem.lower()
    for prefix in ("low", "input", "lq"):
        if lower.startswith(prefix) and len(stem) > len(prefix) and stem[len(prefix)].isdigit():
            return stem[len(prefix):]
    return stem


def _paired_gt_stem_key(stem: str) -> str:
    lower = stem.lower()
    for prefix in ("normal", "high", "gt"):
        if lower.startswith(prefix) and len(stem) > len(prefix) and stem[len(prefix)].isdigit():
            return stem[len(prefix):]
    return stem


def _match_paired_image_stems(
    lows: Dict[str, str],
    gts: Dict[str, str],
    strict_pairs: bool,
) -> List[Tuple[str, str, str]]:
    exact = sorted(set(lows).intersection(gts))
    pairs = [(key, lows[key], gts[key]) for key in exact]

    unmatched_lows = set(lows) - set(exact)
    unmatched_gts = set(gts) - set(exact)
    gt_by_key: Dict[str, List[str]] = {}
    for stem in unmatched_gts:
        gt_by_key.setdefault(_paired_gt_stem_key(stem), []).append(stem)

    used_gts = set()
    missing_gt = []
    for low_stem in sorted(unmatched_lows):
        low_key = _paired_low_stem_key(low_stem)
        candidates = [
            stem
            for stem in sorted(gt_by_key.get(low_key, []))
            if stem not in used_gts
        ]
        if len(candidates) == 1:
            gt_stem = candidates[0]
            used_gts.add(gt_stem)
            pairs.append((low_key, lows[low_stem], gts[gt_stem]))
        else:
            missing_gt.append(low_stem)

    missing_low = sorted(unmatched_gts - used_gts)
    if strict_pairs and (missing_gt or missing_low):
        raise ValueError(
            "Filename mismatch between low/gt folders after LOL stem normalization: "
            f"missing_gt={missing_gt[:3]} missing_low={missing_low[:3]}"
        )

    return sorted(pairs, key=lambda item: item[0])

class DDRTrainDataset(Dataset):
    """
    Returns (low_img, gt_img) pairs from a folder + a JSON (pandas df saved with orient="records").

    Changes vs original:
      - If patch_size != -1: DO NOT resize. Load full image, then take a random crop of size
        patch_size x patch_size (paired between low & gt).
      - If patch_size == -1: return original image (no crop/resize).

    Keeps everything else the same.
    """

    def __init__(
        self,
        data_root: str,
        patch_size: int,
        crop_h: Optional[int] = None,
        crop_w: Optional[int] = None,
        low_subdir: Optional[str] = None,
        gt_subdir: Optional[str] = None,
        return_uint8: bool = False,
        use_aug: bool = False,
        transform: Optional[Callable] = None,
        return_paths: bool = False,
        preprocess_file_path="",
        pad_to_patch_size: bool = False,
        crop_padding_mode: str = "reflect",
        augment: Optional[Dict[str, Any]] = None,
        random_bucket_crop: Optional[Dict[str, Any]] = None,
    ):
        self.data_root = os.path.join(data_root, "DLL_train_full")
        self.pairs_json_path = os.path.join(os.path.dirname(__file__), "dll_train_metadata.json")
        self.patch_size = int(patch_size)
        self.crop_h = None if crop_h is None else int(crop_h)
        self.crop_w = None if crop_w is None else int(crop_w)
        self.fixed_crop_size = _fixed_crop_size(self.patch_size, self.crop_h, self.crop_w)
        self.return_uint8 = bool(return_uint8)
        self.use_aug = bool(use_aug)
        self.return_paths = bool(return_paths)
        self.preprocess_file_path = preprocess_file_path
        self.pad_to_patch_size = bool(pad_to_patch_size)
        self.crop_padding_mode = str(crop_padding_mode)
        self.augment_config = augment if _is_aug_config_present(augment) else None
        self.standard_aug_enabled = _aug_enabled(self.augment_config, self.use_aug)
        self.resize_aug_enabled = _resize_aug_enabled(self.augment_config)
        self.geometry_config = _geometry_config(self.augment_config)
        self.resize_aug_config = _resize_aug_config(self.augment_config)
        self.lq_only_config = _lq_only_config(self.augment_config) if self.standard_aug_enabled else {"enabled": False}
        self.random_bucket_crop_policy = RandomBucketCropPolicy(
            _random_bucket_crop_cfg(self.augment_config, random_bucket_crop)
        )
        self._shared_patch_size = mp.Value("i", self.patch_size)
        self._transform_cache: Dict[Tuple[int, int], Callable] = {}

        self.low_dir = os.path.join(self.data_root, low_subdir) if low_subdir else self.data_root
        self.gt_dir = os.path.join(self.data_root, gt_subdir) if gt_subdir else self.data_root

        self.pairs = self._load_pairs(self.pairs_json_path)
        self._log_augmentation_config()

        # If user provides a transform, it must be a paired transform: (low, gt) -> (low, gt)
        self.transform = transform or self._build_paired_transform()

    def _log_augmentation_config(self):
        if self.standard_aug_enabled or self.resize_aug_enabled or self.random_bucket_crop_policy.enabled:
            print(
                "[DDRTrainDataset] augmentation enabled: "
                f"standard={self.standard_aug_enabled} resize_aug={self.resize_aug_enabled} "
                f"geometry={self.geometry_config} resize={self.resize_aug_config if self.resize_aug_enabled else None} "
                f"lq_only={self.lq_only_config if self.lq_only_config.get('enabled', False) else None} "
                f"random_bucket_crop={self.random_bucket_crop_policy.describe() if self.random_bucket_crop_policy.enabled else None}",
                flush=True,
            )

    @staticmethod
    def _load_pairs(json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("pairs JSON must be a list of records (pandas orient='records').")

        pairs = []
        for i, rec in enumerate(data):
            if not isinstance(rec, dict) or "low" not in rec or "gt" not in rec:
                raise ValueError(f"Invalid record at index {i}: {rec}")
            pairs.append((rec["low"], rec["gt"]))
        return pairs

    def _build_paired_transform(self, crop_size: Optional[Tuple[int, int]] = None) -> Callable:
        padder = None
        cropper = None
        crop_size = crop_size or self.fixed_crop_size

        # Keep original size if no crop is configured.
        if crop_size is not None:
            if self.pad_to_patch_size or self.random_bucket_crop_policy.enabled:
                padder = PairedPadToSize(crop_size, mode=self.crop_padding_mode)
            if self.standard_aug_enabled or self.resize_aug_enabled or self.random_bucket_crop_policy.enabled:
                cropper = PairedRandomCrop(crop_size)
            else:
                cropper = v2.CenterCrop(crop_size)

        # Output format
        ops = []
        if self.return_uint8:
            ops.extend([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
            ])
        else:
            ops.append(v2.ToTensor())

        composed = v2.Compose(ops)

        geometry = PairedGeometryAugment(**self.geometry_config) if self.standard_aug_enabled else None
        lq_only = LQOnlyAugment(self.lq_only_config)

        def paired(low_img, gt_img):
            if self.resize_aug_enabled:
                low_img, gt_img = _paired_resize_aug(low_img, gt_img, self.resize_aug_config)
            if padder is not None:
                low_img, gt_img = padder(low_img, gt_img)
            if cropper is not None:
                low_img, gt_img = cropper(low_img, gt_img)
            low_img, gt_img = composed(low_img, gt_img)
            if geometry is not None:
                low_img, gt_img = geometry(low_img, gt_img)
            low_img = lq_only(low_img)
            return low_img, gt_img

        return paired

    @property
    def current_patch_size(self) -> int:
        return int(self._shared_patch_size.value)

    def set_patch_size(self, patch_size: int) -> None:
        patch_size = int(patch_size)
        self.patch_size = patch_size
        if self.crop_h is None and self.crop_w is None:
            self.fixed_crop_size = _fixed_crop_size(self.patch_size, None, None)
        self._shared_patch_size.value = patch_size
        self.transform = self._build_paired_transform()

    def _transform_for_current_patch(self, crop_size: Optional[Tuple[int, int]] = None) -> Callable:
        if crop_size is None and self.random_bucket_crop_policy.enabled:
            crop_size = self.random_bucket_crop_policy.sample_shape()
        if crop_size is not None:
            crop_key = (int(crop_size[0]), int(crop_size[1]))
            transform = self._transform_cache.get(crop_key)
            if transform is None:
                transform = self._build_paired_transform(crop_key)
                self._transform_cache[crop_key] = transform
            return transform
        patch_size = self.current_patch_size
        fixed_crop = _fixed_crop_size(patch_size, self.crop_h, self.crop_w)
        crop_key = fixed_crop or (-1, -1)
        transform = self._transform_cache.get(crop_key)
        if transform is None:
            self.patch_size = patch_size
            self.fixed_crop_size = fixed_crop
            transform = self._build_paired_transform(fixed_crop)
            self._transform_cache[crop_key] = transform
        return transform

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _resolve_path(base_dir: str, fname: str) -> str:
        return fname if os.path.isabs(fname) else os.path.join(base_dir, fname)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, idx: int):
        idx, crop_size = _unpack_bucket_index(idx)
        low_name, gt_name = self.pairs[idx]

        low_path = self._resolve_path(self.low_dir, low_name)
        gt_path = self._resolve_path(self.gt_dir, gt_name)

        if not os.path.exists(low_path):
            raise FileNotFoundError(f"LOW image not found: {low_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT image not found: {gt_path}")

        if self.preprocess_file_path != "":
            _, file_name = os.path.split(low_path)
            low_path = os.path.join(self.preprocess_file_path, 'train', file_name)

        low_img = self._load_rgb(low_path)
        gt_img = self._load_rgb(gt_path)

        low_img, gt_img = self._transform_for_current_patch(crop_size)(low_img, gt_img)

        if self.return_paths:
            return low_img, gt_img, low_path, gt_path
        return low_img, gt_img

class DDRValDataset(Dataset):
    """
    Validation dataset for paired LOW / GT images.

    Directory structure:
        base_path/
          low/<im_id>.jpg
          gt/<im_id>.jpg

    Args:
        base_path: Root path containing `low/` and `gt/`
        patch_size:
            - -1  -> return original resolution
            - >0  -> resize shorter side to patch_size and center-crop
        return_uint8:
            - True  -> uint8 tensor [0..255]
            - False -> float32 tensor [0..1]
        extensions: allowed image extensions
        return_paths: if True, also return image paths
    """

    def __init__(
        self,
        base_path: str,
        patch_size: int,
        return_uint8: bool = False,
        extensions: Optional[List[str]] = None,
        return_paths: bool = False,
        preprocess_file_path="",
    ):
        self.base_path = os.path.join(base_path, "DLL_val")
        self.low_dir = os.path.join(self.base_path, "low")
        self.gt_dir = os.path.join(self.base_path, "gt")
        self.preprocess_file_path = preprocess_file_path if preprocess_file_path is not None else ""

        if not os.path.isdir(self.low_dir) or not os.path.isdir(self.gt_dir):
            raise ValueError("Expected subfolders: base_path/low and base_path/gt")

        self.patch_size = int(patch_size)
        self.return_uint8 = bool(return_uint8)
        self.return_paths = bool(return_paths)

        self.extensions = extensions or [".jpg", ".jpeg", ".png"]

        self.pairs = self._collect_pairs()
        self.transform = self._build_transform()

    def _collect_pairs(self) -> List[Tuple[str, str, str]]:
        gt_by_stem = {}
        for fname in sorted(os.listdir(self.gt_dir)):
            name, ext = os.path.splitext(fname)
            if ext.lower() in self.extensions:
                path = os.path.join(self.gt_dir, fname)
                if os.path.isfile(path):
                    gt_by_stem.setdefault(name, path)

        pairs = []
        for fname in sorted(os.listdir(self.low_dir)):
            name, ext = os.path.splitext(fname)
            if ext.lower() not in self.extensions:
                continue

            low_path = os.path.join(self.low_dir, fname)
            gt_path = gt_by_stem.get(name)

            if os.path.isfile(low_path) and gt_path is not None:
                pairs.append((low_path, gt_path, name))

        if not pairs:
            raise RuntimeError("No valid LOW/GT pairs found")

        return pairs

    def _build_transform(self):
        ops = []

        if self.patch_size != -1:
            ops.extend([
                v2.Resize(self.patch_size, interpolation=v2.InterpolationMode.BICUBIC),
                # v2.CenterCrop(self.patch_size),
            ])

        if self.return_uint8:
            ops.extend([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
            ])
        else:
            ops.append(v2.ToTensor())

        return v2.Compose(ops)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, gt_path, im_id = self.pairs[idx]

        if self.preprocess_file_path != "":
            _, file_name = os.path.split(low_path)
            low_path = os.path.join(self.preprocess_file_path, 'val', file_name)
            if 'swinir' in self.preprocess_file_path:
                low_path = low_path.replace('.jpg', '.png')

        low_img = self._load_rgb(low_path)
        gt_img = self._load_rgb(gt_path)

        low_img = self.transform(low_img)
        gt_img = self.transform(gt_img)

        if self.return_paths:
            return low_img, gt_img, im_id

        return low_img, gt_img


class PairedImageTrainDataset(Dataset):
    """
    Generic paired low/GT training dataset. Image files are matched by stem,
    so low/foo.png can pair with high/foo.jpg.
    """

    def __init__(
        self,
        data_root: str,
        patch_size: int,
        crop_h: Optional[int] = None,
        crop_w: Optional[int] = None,
        low_dir: str = "low",
        gt_dir: str = "gt",
        return_uint8: bool = False,
        use_aug: bool = False,
        extensions: Optional[List[str]] = None,
        return_paths: bool = False,
        strict_pairs: bool = True,
        pad_to_patch_size: bool = False,
        crop_padding_mode: str = "reflect",
        augment: Optional[Dict[str, Any]] = None,
        random_bucket_crop: Optional[Dict[str, Any]] = None,
    ):
        self.data_root = data_root
        self.low_dir = low_dir if os.path.isabs(low_dir) else os.path.join(data_root, low_dir)
        self.gt_dir = gt_dir if os.path.isabs(gt_dir) else os.path.join(data_root, gt_dir)
        self.patch_size = int(patch_size)
        self.crop_h = None if crop_h is None else int(crop_h)
        self.crop_w = None if crop_w is None else int(crop_w)
        self.fixed_crop_size = _fixed_crop_size(self.patch_size, self.crop_h, self.crop_w)
        self.return_uint8 = bool(return_uint8)
        self.use_aug = bool(use_aug)
        self.return_paths = bool(return_paths)
        self.strict_pairs = bool(strict_pairs)
        self.pad_to_patch_size = bool(pad_to_patch_size)
        self.crop_padding_mode = str(crop_padding_mode)
        self.augment_config = augment if _is_aug_config_present(augment) else None
        self.standard_aug_enabled = _aug_enabled(self.augment_config, self.use_aug)
        self.resize_aug_enabled = _resize_aug_enabled(self.augment_config)
        self.geometry_config = _geometry_config(self.augment_config)
        self.resize_aug_config = _resize_aug_config(self.augment_config)
        self.lq_only_config = _lq_only_config(self.augment_config) if self.standard_aug_enabled else {"enabled": False}
        self.random_bucket_crop_policy = RandomBucketCropPolicy(
            _random_bucket_crop_cfg(self.augment_config, random_bucket_crop)
        )
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp"]
        self._shared_patch_size = mp.Value("i", self.patch_size)
        self._transform_cache: Dict[Tuple[int, int], Callable] = {}

        if not os.path.isdir(self.low_dir):
            raise FileNotFoundError(f"LOW dir not found: {self.low_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"GT dir not found: {self.gt_dir}")

        self.pairs = self._build_pairs()
        self._log_augmentation_config()
        self.transform = self._build_paired_transform()

    def _log_augmentation_config(self):
        if self.standard_aug_enabled or self.resize_aug_enabled or self.random_bucket_crop_policy.enabled:
            print(
                "[PairedImageTrainDataset] augmentation enabled: "
                f"standard={self.standard_aug_enabled} resize_aug={self.resize_aug_enabled} "
                f"geometry={self.geometry_config} resize={self.resize_aug_config if self.resize_aug_enabled else None} "
                f"lq_only={self.lq_only_config if self.lq_only_config.get('enabled', False) else None} "
                f"random_bucket_crop={self.random_bucket_crop_policy.describe() if self.random_bucket_crop_policy.enabled else None}",
                flush=True,
            )

    def _list_images_by_stem(self, folder: str) -> Dict[str, str]:
        out = {}
        for fname in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in self.extensions:
                continue
            path = os.path.join(folder, fname)
            if os.path.isfile(path):
                out.setdefault(stem, path)
        return out

    def _build_pairs(self) -> List[Tuple[str, str]]:
        lows = self._list_images_by_stem(self.low_dir)
        gts = self._list_images_by_stem(self.gt_dir)
        pairs = _match_paired_image_stems(lows, gts, self.strict_pairs)
        if not pairs:
            raise RuntimeError(f"No paired images found in {self.low_dir} and {self.gt_dir}")
        return [(low_path, gt_path) for _, low_path, gt_path in pairs]

    def _build_paired_transform(self, crop_size: Optional[Tuple[int, int]] = None) -> Callable:
        padder = None
        cropper = None
        crop_size = crop_size or self.fixed_crop_size
        if crop_size is not None:
            if self.pad_to_patch_size or self.random_bucket_crop_policy.enabled:
                padder = PairedPadToSize(crop_size, mode=self.crop_padding_mode)
            if self.standard_aug_enabled or self.resize_aug_enabled or self.random_bucket_crop_policy.enabled:
                cropper = PairedRandomCrop(crop_size)
            else:
                cropper = v2.CenterCrop(crop_size)

        ops = []
        if self.return_uint8:
            ops.extend([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])
        else:
            ops.append(v2.ToTensor())

        composed = v2.Compose(ops)
        geometry = PairedGeometryAugment(**self.geometry_config) if self.standard_aug_enabled else None
        lq_only = LQOnlyAugment(self.lq_only_config)

        def paired(low_img, gt_img):
            if self.resize_aug_enabled:
                low_img, gt_img = _paired_resize_aug(low_img, gt_img, self.resize_aug_config)
            if padder is not None:
                low_img, gt_img = padder(low_img, gt_img)
            if cropper is not None:
                low_img, gt_img = cropper(low_img, gt_img)
            low_img, gt_img = composed(low_img, gt_img)
            if geometry is not None:
                low_img, gt_img = geometry(low_img, gt_img)
            low_img = lq_only(low_img)
            return low_img, gt_img

        return paired

    @property
    def current_patch_size(self) -> int:
        return int(self._shared_patch_size.value)

    def set_patch_size(self, patch_size: int) -> None:
        patch_size = int(patch_size)
        self.patch_size = patch_size
        if self.crop_h is None and self.crop_w is None:
            self.fixed_crop_size = _fixed_crop_size(self.patch_size, None, None)
        self._shared_patch_size.value = patch_size
        self.transform = self._build_paired_transform()

    def _transform_for_current_patch(self, crop_size: Optional[Tuple[int, int]] = None) -> Callable:
        if crop_size is None and self.random_bucket_crop_policy.enabled:
            crop_size = self.random_bucket_crop_policy.sample_shape()
        if crop_size is not None:
            crop_key = (int(crop_size[0]), int(crop_size[1]))
            transform = self._transform_cache.get(crop_key)
            if transform is None:
                transform = self._build_paired_transform(crop_key)
                self._transform_cache[crop_key] = transform
            return transform
        patch_size = self.current_patch_size
        fixed_crop = _fixed_crop_size(patch_size, self.crop_h, self.crop_w)
        crop_key = fixed_crop or (-1, -1)
        transform = self._transform_cache.get(crop_key)
        if transform is None:
            self.patch_size = patch_size
            self.fixed_crop_size = fixed_crop
            transform = self._build_paired_transform(fixed_crop)
            self._transform_cache[crop_key] = transform
        return transform

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        idx, crop_size = _unpack_bucket_index(idx)
        low_path, gt_path = self.pairs[idx]
        low_img = self._load_rgb(low_path)
        gt_img = self._load_rgb(gt_path)
        low_img, gt_img = self._transform_for_current_patch(crop_size)(low_img, gt_img)
        if self.return_paths:
            return low_img, gt_img, low_path, gt_path
        return low_img, gt_img


class PairedImageValDataset(Dataset):
    """
    Generic paired validation/evaluation dataset. Image files are matched by stem.
    """

    def __init__(
        self,
        data_root: str,
        patch_size: int,
        low_dir: str = "low",
        gt_dir: str = "gt",
        return_uint8: bool = False,
        extensions: Optional[List[str]] = None,
        return_paths: bool = False,
        strict_pairs: bool = True,
    ):
        self.data_root = data_root
        self.low_dir = low_dir if os.path.isabs(low_dir) else os.path.join(data_root, low_dir)
        self.gt_dir = gt_dir if os.path.isabs(gt_dir) else os.path.join(data_root, gt_dir)
        self.patch_size = int(patch_size)
        self.return_uint8 = bool(return_uint8)
        self.return_paths = bool(return_paths)
        self.strict_pairs = bool(strict_pairs)
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp"]

        if not os.path.isdir(self.low_dir):
            raise FileNotFoundError(f"LOW dir not found: {self.low_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"GT dir not found: {self.gt_dir}")

        self.pairs = self._build_pairs()
        self.transform = self._build_transform()

    def _list_images_by_stem(self, folder: str) -> Dict[str, str]:
        out = {}
        for fname in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in self.extensions:
                continue
            path = os.path.join(folder, fname)
            if os.path.isfile(path):
                out.setdefault(stem, path)
        return out

    def _build_pairs(self) -> List[Tuple[str, str, str]]:
        lows = self._list_images_by_stem(self.low_dir)
        gts = self._list_images_by_stem(self.gt_dir)
        pairs = _match_paired_image_stems(lows, gts, self.strict_pairs)
        if not pairs:
            raise RuntimeError(f"No paired images found in {self.low_dir} and {self.gt_dir}")
        return [(low_path, gt_path, key) for key, low_path, gt_path in pairs]

    def _build_transform(self):
        ops = []
        if self.patch_size != -1:
            ops.append(v2.Resize(self.patch_size, interpolation=v2.InterpolationMode.BICUBIC))
        if self.return_uint8:
            ops.extend([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])
        else:
            ops.append(v2.ToTensor())
        return v2.Compose(ops)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        low_path, gt_path, im_id = self.pairs[idx]
        low_img = self.transform(self._load_rgb(low_path))
        gt_img = self.transform(self._load_rgb(gt_path))
        if self.return_paths:
            return low_img, gt_img, im_id
        return low_img, gt_img



class DDRTrainPatchesDataset(Dataset):
    """
    Dataset for the *patches* variant:

      <data_path>/high/<file_name>.png
      <data_path>/low/<file_name>.png

    - No metadata JSON needed: pairs are inferred by matching filenames.
    - Length is number of items in one folder (by default: intersection of low & high).
    - Data is already 512x512 patches.
      * Default: no cropping (returns full patch).
      * If patch_size != -1 and patch_size < 512: optional crop (random if use_aug else center).
      * If patch_size == -1: return full patch (no crop).
      * If patch_size > 512: raises ValueError (doesn't make sense for this dataset).

    Augmentations:
      - Horizontal flip (optional)
      - Rot90 (optional, via your paired_rot90)
      - Cropping only when patch_size < 512 (optional)
    """

    def __init__(
        self,
        data_root: str,                 # points directly to the patches dataset root
        patch_size: int = -1,           # default: no crop (full 512)
        low_subdir: str = "low",
        gt_subdir: str = "high",
        return_uint8: bool = False,
        use_aug: bool = False,
        transform: Optional[Callable] = None,   # paired transform: (low, gt)->(low, gt)
        return_paths: bool = False,
        strict_pairs: bool = True,      # if True: require exact filename match in both dirs
    ):
        self.data_root = os.path.join(data_root, 'DLL_train_patches512')
        self.low_dir = os.path.join(self.data_root, low_subdir)
        self.gt_dir = os.path.join(self.data_root, gt_subdir)

        self.patch_size = int(patch_size)
        self.return_uint8 = bool(return_uint8)
        self.use_aug = bool(use_aug)
        self.return_paths = bool(return_paths)
        self.strict_pairs = bool(strict_pairs)

        if not os.path.isdir(self.low_dir):
            raise FileNotFoundError(f"LOW dir not found: {self.low_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"GT/HIGH dir not found: {self.gt_dir}")

        # Build list of paired filenames
        self.pairs: List[Tuple[str, str]] = self._build_pairs()

        # If user provides a transform, it must be paired: (low, gt) -> (low, gt)
        self.transform = transform or self._build_paired_transform()

    def _build_pairs(self) -> List[Tuple[str, str]]:
        def list_pngs(d: str) -> List[str]:
            return sorted(
                [f for f in os.listdir(d) if f.lower().endswith(".png") and os.path.isfile(os.path.join(d, f))]
            )

        low_files = list_pngs(self.low_dir)
        gt_files = list_pngs(self.gt_dir)

        low_set = set(low_files)
        gt_set = set(gt_files)

        if self.strict_pairs:
            common = sorted(low_set.intersection(gt_set))
            missing_low = sorted(gt_set - low_set)
            missing_gt = sorted(low_set - gt_set)

            if missing_low or missing_gt:
                # Fail fast: mismatched dataset is usually a bug
                msg = []
                if missing_low:
                    msg.append(f"{len(missing_low)} files exist in high but missing in low (e.g. {missing_low[:3]})")
                if missing_gt:
                    msg.append(f"{len(missing_gt)} files exist in low but missing in high (e.g. {missing_gt[:3]})")
                raise ValueError("Filename mismatch between low/high folders: " + " | ".join(msg))

            return [(os.path.join(self.low_dir, f), os.path.join(self.gt_dir, f)) for f in common]

        # Non-strict: pair by intersection and ignore extras
        common = sorted(low_set.intersection(gt_set))
        return [(os.path.join(self.low_dir, f), os.path.join(self.gt_dir, f)) for f in common]

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _build_paired_transform(self) -> Callable:
        ops = []

        # Cropping logic: patches are 512x512 by construction.
        # Default (patch_size == -1): no crop.
        # If patch_size < 512: crop. If use_aug: random crop else center crop.
        if self.patch_size != -1:
            if self.patch_size > 512:
                raise ValueError(f"patch_size={self.patch_size} > 512 is not supported for 512x512 patches.")
            if self.patch_size < 512:
                if self.use_aug:
                    ops.extend([
                        v2.RandomHorizontalFlip(p=0.5),
                        v2.RandomCrop((self.patch_size, self.patch_size)),
                    ])
                else:
                    ops.append(v2.CenterCrop((self.patch_size, self.patch_size)))
            else:
                # Full-size patches skip cropping and only keep flip augmentation.
                if self.use_aug:
                    ops.append(v2.RandomHorizontalFlip(p=0.5))
        else:
            # Disabled patch cropping still keeps flip augmentation when enabled.
            if self.use_aug:
                ops.append(v2.RandomHorizontalFlip(p=0.5))

        # Output format
        if self.return_uint8:
            ops.extend([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])
        else:
            ops.append(v2.ToTensor())

        composed = v2.Compose(ops)

        # Paired wrapper: v2 can take multiple inputs and share RNG params
        def paired(low_img, gt_img):
            low_img, gt_img = composed(low_img, gt_img)
            return low_img, gt_img

        return paired

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        low_path, gt_path = self.pairs[idx]

        if not os.path.exists(low_path):
            raise FileNotFoundError(f"LOW image not found: {low_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT/HIGH image not found: {gt_path}")

        low_img = self._load_rgb(low_path)
        gt_img = self._load_rgb(gt_path)

        low_img, gt_img = self.transform(low_img, gt_img)

        # Keep your existing paired rot90 augmentation
        if self.use_aug:
            low_img, gt_img = paired_rot90(low_img, gt_img, p=0.5)

        if self.return_paths:
            return low_img, gt_img, low_path, gt_path
        return low_img, gt_img
# ---- Example usage ----
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_path = "/workspace/datasets/DLL_dataset"
    patch_size = 1024
    ds = DDRTrainDataset(
        data_root=data_path,
        patch_size=patch_size,
        use_aug=True,
    )

    val_dset = DDRValDataset(
        base_path="/workspace/datasets/DLL_val",
        patch_size=-1,
        return_uint8=False,
    )

    low, gt = val_dset[0]


    plt.subplot(1, 2, 1)
    plt.imshow(low.permute(1, 2, 0).numpy())
    plt.title("Low-light")
    plt.subplot(1, 2, 2)
    plt.imshow(gt.permute(1, 2, 0).numpy())
    plt.title("Ground-truth")
    plt.show()


    quit()
