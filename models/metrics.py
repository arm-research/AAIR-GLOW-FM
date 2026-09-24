# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from PIL import Image
import pyiqa
import torch.distributed as dist

VERBOSE = False

GT_MEAN_MATCH_METRICS = {"psnr", "ssim", "lpips", "lpips_alex", "dists"}
NO_REFERENCE_METRICS = {"niqe", "brisque", "clipiqa", "musiq", "liqe", "qalign"}
MIN_SHORT_SIDE_METRICS = {"liqe", "qalign"}
MEAN_ERROR_METRICS = {
    "global_mean_signed_err",
    "channel_mean_signed_err",
    "r_mean_signed_err",
    "g_mean_signed_err",
    "b_mean_signed_err",
}


def match_prediction_mean_to_gt(x_hq_hat, x_hq, eps=1e-6):
    """Scale each prediction so its image mean matches the paired GT mean."""
    if x_hq is None:
        raise ValueError("GT mean matching requires paired ground-truth images.")
    pred_mean = x_hq_hat.mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    gt_mean = x_hq.mean(dim=(1, 2, 3), keepdim=True)
    return torch.clamp(x_hq_hat * (gt_mean / pred_mean), 0.0, 1.0)


def signed_mean_error_values(x_hq_hat, x_hq):
    """Return per-image signed mean errors between prediction and GT."""
    if x_hq is None:
        raise ValueError("Signed mean errors require paired ground-truth images.")
    if x_hq_hat.shape != x_hq.shape:
        raise ValueError(
            f"Signed mean errors require matching shapes, got {tuple(x_hq_hat.shape)} and {tuple(x_hq.shape)}"
        )
    err_global = x_hq_hat.float().mean(dim=(1, 2, 3)) - x_hq.float().mean(dim=(1, 2, 3))
    err_channel = x_hq_hat.float().mean(dim=(2, 3)) - x_hq.float().mean(dim=(2, 3))
    values = {
        "global_mean_signed_err": err_global,
        "channel_mean_signed_err": err_channel.mean(dim=1),
    }
    if err_channel.shape[1] >= 3:
        values.update(
            {
                "r_mean_signed_err": err_channel[:, 0],
                "g_mean_signed_err": err_channel[:, 1],
                "b_mean_signed_err": err_channel[:, 2],
            }
        )
    return values

def my_save_img(img, path):
    if img.dtype != torch.uint8:
        save_image(img, path)
    else:
        if img.is_cuda:
            img = img.cpu()

        img = img.permute(1, 2, 0)
        img = img.numpy()
        Image.fromarray(img).save(path)

def ddp_is_on():
    return dist.is_available() and dist.is_initialized()



class MetricEval(object):
    def __init__(self, metric, device, out_dir=None, keep_saved=False, lpips_net="vgg"):
        self.metric = metric
        self.values = []
        self.count = 0
        self.image_size = 512
        self.out_dir = out_dir
        self.keep_saved = keep_saved
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.lpips_net = lpips_net

        # DDP-safe accumulators
        self.sum = torch.tensor(0.0, device=device)
        self.n = torch.tensor(0.0, device=device)

        if metric in MEAN_ERROR_METRICS:
            self.evaluater = None
        elif metric=="fid-g":
            # FID vs ground-truth
            self.evaluater = FrechetInceptionDistance(normalize=True, reset_real_features=False)
            load_fid_statistics(os.path.join(os.path.dirname(__file__),"datasets","celeba_fid_stat.pt"), self.evaluater, device)
            self.evaluater.to(device)
        elif metric == "fid-f":
            # FID vs FFHQ dataset
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir,"out")
            if self.rank == 0:
                shutil.rmtree(self.out_dir, ignore_errors=True)
            if ddp_is_on():
                dist.barrier()  # ensure folder state is consistent
            os.makedirs(self.out_dir, exist_ok=True)

            self.evaluater = pyiqa.create_metric('fid', device=device)
        elif metric == "lpips":
            self.evaluater = pyiqa.create_metric(metric, device=device, net=self.lpips_net)
        elif metric == "lpips_alex":
            self.evaluater = pyiqa.create_metric('lpips', device=device, net='alex')
        elif metric == "save":
            # save images
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir,"out")
            # shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
        else:
            self.evaluater = pyiqa.create_metric(metric, device=device)

    def _get_metric_mode(self):
        return getattr(self.evaluater, "metric_mode", None)

    def _get_metric_device(self):
        metric_device = getattr(self.evaluater, "device", None)
        if metric_device is None:
            metric_device = self.sum.device
        return torch.device(metric_device)

    def compute(self, x_hq_hat, x_hq=None):
        bs = x_hq_hat.shape[0]
        if self.metric in MEAN_ERROR_METRICS:
            values = signed_mean_error_values(x_hq_hat, x_hq)
            if self.metric not in values:
                raise ValueError(f"Metric {self.metric} requires an RGB tensor with at least 3 channels")
            value = values[self.metric]
            self.count += bs
        elif self.metric=="fid-g":
            self.evaluater.to(x_hq_hat.device)
            self.evaluater.update(x_hq_hat, real=False)
            value = torch.zeros((len(x_hq_hat),), device=x_hq_hat.device)
            self.count += bs
        elif self.metric=="fid-f":
            self.image_size = x_hq.shape[-1]
            for img in x_hq_hat:
                my_save_img(img, os.path.join(self.out_dir, f"rank_{self.rank}_{self.count:08d}.png"))
                self.count += 1
            value = torch.zeros((len(x_hq_hat),), device=x_hq_hat.device)
        elif self.metric in NO_REFERENCE_METRICS:
            # Debug Print
            #Chceck for nan or inf values
            if torch.isnan(x_hq_hat).any() or torch.isinf(x_hq_hat).any():
                print(f"Warning: Input contains NaN or Inf values on rank {self.rank}. Replacing with zeros.")
                x_hq_hat = torch.nan_to_num(x_hq_hat, nan=0.0, posinf=1.0, neginf=0.0)
            if self.metric in MIN_SHORT_SIDE_METRICS and min(x_hq_hat.shape[-2:]) < 224:
                x_hq_hat = F.interpolate(
                    x_hq_hat,
                    size=(224, 224),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp(0.0, 1.0)
            try:
                value = self.evaluater(x_hq_hat)
            except Exception as exc:
                if self.metric not in {"niqe", "musiq"}:
                    raise
                print(
                    f"Warning: {self.metric} failed on rank {self.rank}: {type(exc).__name__}: {exc}. "
                    "Logging NaN for this batch and continuing.",
                    flush=True,
                )
                value = torch.full((bs,), float("nan"), device=self.sum.device)
            self.count += bs
        elif self.metric == "save":
            for img in x_hq_hat:
                my_save_img(img, os.path.join(self.out_dir, f"rank_{self.rank}_{self.count:08d}.png"))
                self.count += 1
            value = torch.zeros((len(x_hq_hat),))
        else:
            metric_mode = self._get_metric_mode()
            metric_device = self._get_metric_device()
            x_hq_hat_metric = x_hq_hat.to(metric_device)

            # Guard NR metrics against invalid values before handing them to pyiqa.
            if metric_mode == "NR" and (torch.isnan(x_hq_hat_metric).any() or torch.isinf(x_hq_hat_metric).any()):
                print(f"Warning: Input contains NaN or Inf values on rank {self.rank}. Replacing with zeros.")
                x_hq_hat_metric = torch.nan_to_num(x_hq_hat_metric, nan=0.0, posinf=1.0, neginf=0.0)

            if metric_mode == "FR":
                assert x_hq is not None, f"Metric {self.metric} requires ground-truth images."
                assert x_hq_hat.shape == x_hq.shape, (
                    f"Metric {self.metric} is expected to y and y_hat to be the same size! "
                    "Make sure you use dataset with labels."
                )
                value = self.evaluater(x_hq_hat_metric, x_hq.to(metric_device))
            elif metric_mode == "NR":
                value = self.evaluater(x_hq_hat_metric)
            else:
                raise ValueError(f"Unsupported metric mode for {self.metric}: {metric_mode!r}")
            self.count += bs
        # self.values.append(value.reshape(-1))
        if torch.is_tensor(value):
            value = value.to(self.sum.device)
        else:
            value = torch.tensor(value, device=self.sum.device)
        value = value.reshape(-1)
        finite_value = value[torch.isfinite(value)]
        if finite_value.numel() > 0:
            self.sum += finite_value.sum()
            self.n += float(finite_value.numel())

    def get_final(self):
        if self.metric=="fid-g":
            final_value = self.evaluater.compute()
            self.evaluater.reset()
        elif self.metric=="fid-f":
            if ddp_is_on():
                dist.barrier()  # wait for all ranks to finish writing

            if self.rank == 0:
                final_value = self.evaluater(self.out_dir,
                                             dataset_name="FFHQ",
                                             dataset_res=self.image_size,
                                             dataset_split="trainval70k",
                                             verbose=VERBOSE)
                if not self.keep_saved:
                    shutil.rmtree(self.out_dir, ignore_errors=True)
                    os.makedirs(self.out_dir, exist_ok=True)
            else:
                final_value = torch.tensor(float("nan"), device="cpu")

            if ddp_is_on():
                dist.barrier()  # all ranks wait until rank 0 finished cleanup

            # reset state on ALL ranks
            self.count = 0
            self.values.clear()
            self.sum.zero_()
            self.n.zero_()

            return final_value
        else:
            s = self.sum
            n = self.n
            if ddp_is_on():
                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                dist.all_reduce(n, op=dist.ReduceOp.SUM)
            final_value = torch.where(
                n > 0,
                s / n.clamp_min(1),
                torch.full_like(s, float("nan")),
            )

            self.sum.zero_()
            self.n.zero_()
            self.count = 0
            self.values.clear()  # optional if you stop using it
            return final_value
        self.count = 0
        self.values.clear()
        return final_value

def load_fid_statistics(stat_path, evaluator, device):
    real_features = torch.load(stat_path, map_location=device)
    evaluator.real_features_sum = real_features['real_features_sum']
    evaluator.real_features_cov_sum = real_features['real_features_cov_sum']
    evaluator.real_features_num_samples = real_features['real_features_num_samples']

def save_fid_statistics(evaluater):
    d = {"real_features_sum": evaluater.real_features_sum,
         "real_features_cov_sum": evaluater.real_features_cov_sum,
         "real_features_num_samples": evaluater.real_features_num_samples}
    torch.save(d, "a.pt")
