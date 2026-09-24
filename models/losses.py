# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Loss helpers for low-light flow training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


@dataclass(frozen=True)
class DistortionLossConfig:
    """Configuration for the auxiliary flow reconstruction loss."""

    enabled: bool = False
    weight: float = 0.0
    loss_type: str = "charbonnier"
    reduction: str = "batchmean"
    eps: float = 1e-3


@dataclass(frozen=True)
class RGBReconstructionLossConfig:
    """Configuration for optional RGB-space reconstruction losses."""

    channel_mean_enabled: bool = False
    channel_mean_weight: float = 0.0
    local_patch_mean_enabled: bool = False
    local_patch_mean_weight: float = 0.0
    local_patch_mean_kernel_size: int = 32
    charbonnier_enabled: bool = False
    charbonnier_weight: float = 0.0
    charbonnier_eps: float = 1e-3
    schedule_wait_steps: int = 0
    schedule_ramp_steps: int = 0
    schedule_final_scale: float = 1.0


@dataclass(frozen=True)
class EndpointRGBLossConfig:
    """Configuration for endpoint RGB supervision at the inference condition."""

    enabled: bool = False
    weight: float = 0.1
    prob: float = 0.5
    warmup_steps: int = 5000
    ramp_steps: int = 20000
    eps: float = 1e-3
    bias_time_sampling: bool = True
    bias_prob: float = 0.5
    bias_min: float = 0.8
    force_endpoint_prob: float = 0.0


class DistortionLoss(torch.nn.Module):
    """Penalize distortion in the clean sample reconstructed from a flow field.

    For the rectified-flow path used in ``train.py``, the model predicts the
    velocity from clean target to noisy/low-light prior. Given an intermediate
    latent ``z_t`` and predicted velocity ``v_pred``, the clean prediction is
    ``x0_pred = z_t - t * v_pred``. This loss compares ``x0_pred`` with the
    high-quality latent target.
    """

    _LOSS_TYPES = {"l1", "l2", "mse", "smooth_l1", "charbonnier"}
    _REDUCTIONS = {"none", "mean", "sum", "batchmean"}

    def __init__(
        self,
        enabled: bool = False,
        weight: float = 0.0,
        loss_type: str = "charbonnier",
        reduction: str = "batchmean",
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.loss_type = str(loss_type).lower()
        self.reduction = str(reduction).lower()
        self.eps = float(eps)

        if self.loss_type not in self._LOSS_TYPES:
            raise ValueError(f"Unsupported distortion loss_type: {loss_type}")
        if self.reduction not in self._REDUCTIONS:
            raise ValueError(f"Unsupported distortion reduction: {reduction}")
        if self.weight < 0:
            raise ValueError(f"distortion loss weight must be non-negative, got {weight}")
        if self.eps <= 0:
            raise ValueError(f"distortion loss eps must be positive, got {eps}")

    @property
    def active(self) -> bool:
        """Whether this loss should affect training."""

        return self.enabled and self.weight > 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "DistortionLoss":
        """Build from an OmegaConf/dict config, defaulting to disabled."""

        if cfg is None:
            return cls()
        values = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        if not isinstance(values, Mapping):
            raise TypeError("distortion loss config must be a mapping")
        defaults = DistortionLossConfig()
        return cls(
            enabled=values.get("enabled", defaults.enabled),
            weight=values.get("weight", defaults.weight),
            loss_type=values.get("loss_type", defaults.loss_type),
            reduction=values.get("reduction", defaults.reduction),
            eps=values.get("eps", defaults.eps),
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the weighted distortion loss between prediction and target."""

        self._validate_pair(prediction, target)
        if not self.active:
            return prediction.new_zeros(())

        pred = prediction.float()
        tgt = target.float()
        if self.loss_type == "l1":
            loss_map = (pred - tgt).abs()
        elif self.loss_type in {"l2", "mse"}:
            loss_map = (pred - tgt).square()
        elif self.loss_type == "smooth_l1":
            loss_map = F.smooth_l1_loss(pred, tgt, reduction="none")
        else:
            diff = pred - tgt
            loss_map = torch.sqrt(diff.square() + self.eps * self.eps)

        return self._reduce(loss_map) * self.weight

    def from_flow(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        pred_velocity: torch.Tensor,
        clean_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distortion on ``x0_pred = z_t - t * pred_velocity``."""

        self._validate_pair(z_t, pred_velocity, "z_t", "pred_velocity")
        self._validate_pair(z_t, clean_target, "z_t", "clean_target")
        t = self._broadcast_time(t, z_t)
        x0_pred = z_t.float() - t.float() * pred_velocity.float()
        return self(x0_pred, clean_target.float())

    def _reduce(self, loss_map: torch.Tensor) -> torch.Tensor:
        if self.reduction == "none":
            return loss_map
        if self.reduction == "mean":
            return loss_map.mean()
        if self.reduction == "sum":
            return loss_map.sum()
        return loss_map.flatten(1).mean(dim=1)

    @staticmethod
    def _validate_pair(
        prediction: torch.Tensor,
        target: torch.Tensor,
        prediction_name: str = "prediction",
        target_name: str = "target",
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"{prediction_name} and {target_name} must have matching shapes, "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        if prediction.device != target.device:
            raise ValueError(
                f"{prediction_name} and {target_name} must be on the same device, "
                f"got {prediction.device} and {target.device}"
            )

    @staticmethod
    def _broadcast_time(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        t = t.to(device=reference.device)
        if t.ndim == 1:
            t = t.view(-1, *([1] * (reference.ndim - 1)))
        if t.shape[0] != reference.shape[0]:
            raise ValueError(
                f"time and reference batch sizes must match, got {t.shape[0]} and {reference.shape[0]}"
            )
        return t


class RGBReconstructionLosses(torch.nn.Module):
    """Optional image-space losses for exposure and RGB fidelity probes."""

    def __init__(
        self,
        *,
        channel_mean_enabled: bool = False,
        channel_mean_weight: float = 0.0,
        local_patch_mean_enabled: bool = False,
        local_patch_mean_weight: float = 0.0,
        local_patch_mean_kernel_size: int = 32,
        charbonnier_enabled: bool = False,
        charbonnier_weight: float = 0.0,
        charbonnier_eps: float = 1e-3,
        schedule_wait_steps: int = 0,
        schedule_ramp_steps: int = 0,
        schedule_final_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.channel_mean_enabled = bool(channel_mean_enabled)
        self.channel_mean_weight = float(channel_mean_weight)
        self.local_patch_mean_enabled = bool(local_patch_mean_enabled)
        self.local_patch_mean_weight = float(local_patch_mean_weight)
        self.local_patch_mean_kernel_size = int(local_patch_mean_kernel_size)
        self.charbonnier_enabled = bool(charbonnier_enabled)
        self.charbonnier_weight = float(charbonnier_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.schedule_wait_steps = int(schedule_wait_steps)
        self.schedule_ramp_steps = int(schedule_ramp_steps)
        self.schedule_final_scale = float(schedule_final_scale)

        for name, weight in {
            "channel_mean.weight": self.channel_mean_weight,
            "local_patch_mean.weight": self.local_patch_mean_weight,
            "charbonnier_rgb.weight": self.charbonnier_weight,
            "schedule.final_scale": self.schedule_final_scale,
        }.items():
            if weight < 0:
                raise ValueError(f"{name} must be non-negative, got {weight}")
        if self.schedule_wait_steps < 0:
            raise ValueError(f"schedule.wait_steps must be non-negative, got {self.schedule_wait_steps}")
        if self.schedule_ramp_steps < 0:
            raise ValueError(f"schedule.ramp_steps must be non-negative, got {self.schedule_ramp_steps}")
        if self.local_patch_mean_kernel_size <= 0:
            raise ValueError(
                f"local_patch_mean.kernel_size must be positive, got {self.local_patch_mean_kernel_size}"
            )
        if self.charbonnier_eps <= 0:
            raise ValueError(f"charbonnier_rgb.eps must be positive, got {self.charbonnier_eps}")

    @property
    def active(self) -> bool:
        return any(
            [
                self.channel_mean_enabled and self.channel_mean_weight > 0.0,
                self.local_patch_mean_enabled and self.local_patch_mean_weight > 0.0,
                self.charbonnier_enabled and self.charbonnier_weight > 0.0,
            ]
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "RGBReconstructionLosses":
        if cfg is None:
            return cls()
        values = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        if not isinstance(values, Mapping):
            raise TypeError("RGB reconstruction loss config must be a mapping")
        defaults = RGBReconstructionLossConfig()
        channel = dict(values.get("channel_mean", {}) or {})
        local = dict(values.get("local_patch_mean", {}) or {})
        charb = dict(values.get("charbonnier_rgb", {}) or {})
        schedule = dict(values.get("schedule", {}) or {})
        return cls(
            channel_mean_enabled=channel.get("enabled", defaults.channel_mean_enabled),
            channel_mean_weight=channel.get("weight", defaults.channel_mean_weight),
            local_patch_mean_enabled=local.get("enabled", defaults.local_patch_mean_enabled),
            local_patch_mean_weight=local.get("weight", defaults.local_patch_mean_weight),
            local_patch_mean_kernel_size=local.get("kernel_size", defaults.local_patch_mean_kernel_size),
            charbonnier_enabled=charb.get("enabled", defaults.charbonnier_enabled),
            charbonnier_weight=charb.get("weight", defaults.charbonnier_weight),
            charbonnier_eps=charb.get("eps", defaults.charbonnier_eps),
            schedule_wait_steps=schedule.get("wait_steps", values.get("wait_steps", defaults.schedule_wait_steps)),
            schedule_ramp_steps=schedule.get("ramp_steps", values.get("ramp_steps", defaults.schedule_ramp_steps)),
            schedule_final_scale=schedule.get("final_scale", values.get("final_scale", defaults.schedule_final_scale)),
        )

    def schedule_scale(self, step: int, reference: torch.Tensor) -> torch.Tensor:
        step = max(0, int(step))
        if step < self.schedule_wait_steps:
            scale = 0.0
        elif self.schedule_ramp_steps <= 0:
            scale = self.schedule_final_scale
        else:
            ramp_pos = min(step - self.schedule_wait_steps, self.schedule_ramp_steps)
            scale = self.schedule_final_scale * (float(ramp_pos) / float(self.schedule_ramp_steps))
        return reference.new_tensor(scale, dtype=torch.float32)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        DistortionLoss._validate_pair(prediction, target)
        batch = prediction.shape[0]
        total = prediction.new_zeros((batch,), dtype=torch.float32)
        logs = {}
        if not self.active:
            return total, logs

        pred = prediction.float()
        tgt = target.float()
        if self.channel_mean_enabled and self.channel_mean_weight > 0.0:
            loss = (pred.mean(dim=(2, 3)) - tgt.mean(dim=(2, 3))).abs().mean(dim=1)
            weighted = loss * self.channel_mean_weight
            total = total + weighted
            logs["channel_mean_loss"] = weighted.detach()

        if self.local_patch_mean_enabled and self.local_patch_mean_weight > 0.0:
            k = self.local_patch_mean_kernel_size
            kh = min(k, pred.shape[-2])
            kw = min(k, pred.shape[-1])
            pred_pool = F.avg_pool2d(pred, kernel_size=(kh, kw), stride=1)
            tgt_pool = F.avg_pool2d(tgt, kernel_size=(kh, kw), stride=1)
            loss = (pred_pool - tgt_pool).abs().flatten(1).mean(dim=1)
            weighted = loss * self.local_patch_mean_weight
            total = total + weighted
            logs["local_patch_mean_loss"] = weighted.detach()

        if self.charbonnier_enabled and self.charbonnier_weight > 0.0:
            loss = torch.sqrt((pred - tgt).square() + self.charbonnier_eps * self.charbonnier_eps).flatten(1).mean(dim=1)
            weighted = loss * self.charbonnier_weight
            total = total + weighted
            logs["charbonnier_rgb_loss_raw"] = loss.detach()
            logs["charbonnier_rgb_loss_weighted"] = weighted.detach()
            logs["charbonnier_rgb_loss"] = weighted.detach()

        return total, logs


class EndpointRGBLoss(torch.nn.Module):
    """Auxiliary RGB loss for the pure endpoint ``prior -> GT`` condition."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        weight: float = 0.1,
        prob: float = 0.5,
        warmup_steps: int = 5000,
        ramp_steps: int = 20000,
        eps: float = 1e-3,
        bias_time_sampling: bool = True,
        bias_prob: float = 0.5,
        bias_min: float = 0.8,
        force_endpoint_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.prob = float(prob)
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.eps = float(eps)
        self.bias_time_sampling = bool(bias_time_sampling)
        self.bias_prob = float(bias_prob)
        self.bias_min = float(bias_min)
        self.force_endpoint_prob = float(force_endpoint_prob)

        if self.weight < 0:
            raise ValueError(f"endpoint_rgb.weight must be non-negative, got {weight}")
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"endpoint_rgb.prob must be in [0, 1], got {prob}")
        if self.warmup_steps < 0:
            raise ValueError(f"endpoint_rgb.warmup_steps must be non-negative, got {warmup_steps}")
        if self.ramp_steps < 0:
            raise ValueError(f"endpoint_rgb.ramp_steps must be non-negative, got {ramp_steps}")
        if self.eps <= 0:
            raise ValueError(f"endpoint_rgb.eps must be positive, got {eps}")
        if not 0.0 <= self.bias_prob <= 1.0:
            raise ValueError(f"endpoint_rgb.bias_prob must be in [0, 1], got {bias_prob}")
        if not 0.0 <= self.bias_min <= 1.0:
            raise ValueError(f"endpoint_rgb.bias_min must be in [0, 1], got {bias_min}")
        if not 0.0 <= self.force_endpoint_prob <= 1.0:
            raise ValueError(f"endpoint_rgb.force_endpoint_prob must be in [0, 1], got {force_endpoint_prob}")

    @property
    def active(self) -> bool:
        return self.enabled and self.weight > 0.0 and self.prob > 0.0

    @property
    def time_sampling_active(self) -> bool:
        return self.active and self.bias_time_sampling and (
            self.bias_prob > 0.0 or self.force_endpoint_prob > 0.0
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "EndpointRGBLoss":
        if cfg is None:
            return cls()
        values = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        if not isinstance(values, Mapping):
            raise TypeError("endpoint RGB loss config must be a mapping")
        defaults = EndpointRGBLossConfig()
        time_sampling = dict(values.get("time_sampling", {}) or {})
        return cls(
            enabled=values.get("enabled", defaults.enabled),
            weight=values.get("weight", defaults.weight),
            prob=values.get("prob", defaults.prob),
            warmup_steps=values.get("warmup_steps", defaults.warmup_steps),
            ramp_steps=values.get("ramp_steps", defaults.ramp_steps),
            eps=values.get("eps", defaults.eps),
            bias_time_sampling=time_sampling.get(
                "enabled",
                values.get("bias_time_sampling", defaults.bias_time_sampling),
            ),
            bias_prob=time_sampling.get("prob", values.get("bias_prob", defaults.bias_prob)),
            bias_min=time_sampling.get("min_s_prime", values.get("bias_min", defaults.bias_min)),
            force_endpoint_prob=time_sampling.get(
                "force_endpoint_prob",
                values.get("force_endpoint_prob", defaults.force_endpoint_prob),
            ),
        )

    def schedule_scale(self, step: int, reference: torch.Tensor) -> torch.Tensor:
        step = max(0, int(step))
        if step < self.warmup_steps:
            scale = 0.0
        elif self.ramp_steps <= 0:
            scale = 1.0
        else:
            ramp_pos = min(step - self.warmup_steps, self.ramp_steps)
            scale = float(ramp_pos) / float(self.ramp_steps)
        return reference.new_tensor(scale, dtype=torch.float32)

    def should_apply(self, reference: torch.Tensor) -> bool:
        if not self.active:
            return False
        if self.prob >= 1.0:
            return True
        return bool(torch.rand((), device=reference.device).item() < self.prob)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        DistortionLoss._validate_pair(prediction, target)
        diff = prediction.float() - target.float()
        loss = torch.sqrt(diff.square() + self.eps * self.eps).flatten(1).mean(dim=1)
        return loss * self.weight
