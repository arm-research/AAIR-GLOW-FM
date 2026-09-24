# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Evaluate Glow-FM."""

import argparse
import csv
import json
import math
from pathlib import Path

import pyiqa
import torch

from utils.image_enhancement import (
    enhance,
    index_images,
    load_pipeline,
    prediction_path,
    read_image,
    resolve_device,
    write_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file")
    parser.add_argument("--input", type=Path, required=True, help="Low-light image or directory")
    parser.add_argument("--gt", type=Path, help="Paired ground-truth image or directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional directory for predictions, per-image CSV, and summary JSON",
    )
    parser.add_argument(
        "--metrics",
        default="psnr,ssim,niqe",
        help="Comma-separated PyIQA metric names (default: psnr,ssim,niqe)",
    )
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--nfe", type=int, default=1, help="Number of function evaluations")
    parser.add_argument("--sigma", type=float, default=0.0, help="Latent input-noise standard deviation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nfe < 1:
        raise ValueError("--nfe must be at least 1")
    if args.sigma < 0:
        raise ValueError("--sigma must be non-negative")

    # We use PyIQA to compute image quality metrics.
    # The user can specify which metrics to compute via the --metrics. The default metrics are PSNR, SSIM, and NIQE.
    # The code checks that at least one metric is specified and raises an error if not.
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    if not metric_names:
        raise ValueError("--metrics must contain at least one PyIQA metric name")

    device = resolve_device(args.device)
    metrics = {name: pyiqa.create_metric(name, device=device) for name in metric_names}

    # PyIQA declares whether each metric is full-reference (FR) or
    # no-reference (NR). Fail before inference if an FR metric has no GT.
    reference_metrics = [
        name for name, metric in metrics.items() if getattr(metric, "metric_mode", None) == "FR"
    ]
    if args.gt is None and reference_metrics:
        raise ValueError(
            f"--gt is required for full-reference metrics: {', '.join(reference_metrics)}"
        )

    # Relative, extension-free keys make low/GT pairing deterministic while
    # allowing corresponding images to have different file extensions.
    input_root, input_index = index_images(args.input, exclude=args.output)
    gt_index: dict[str, Path] = {}
    if args.gt is not None:
        _, gt_index = index_images(args.gt, exclude=args.output)
        missing_gt = sorted(set(input_index) - set(gt_index))
        extra_gt = sorted(set(gt_index) - set(input_index))
        if missing_gt or extra_gt:
            raise ValueError(
                "Low/GT image keys do not match: "
                f"missing GT={missing_gt[:5]}, extra GT={extra_gt[:5]}"
            )

    network, encoder = load_pipeline(args.checkpoint, device)
    sums = {name: 0.0 for name in metric_names}
    counts = {name: 0 for name in metric_names}
    rows: list[dict] = []

    for index, (key, input_path) in enumerate(input_index.items(), start=1):
        image = read_image(input_path, device)
        prediction = enhance(
            network,
            encoder,
            image,
            nfe=args.nfe,
            sigma=args.sigma,
        )

        # If no output directory is specified, the enhanced image is not saved to disk.
        output_path = None
        if args.output is not None:
            output_path = prediction_path(input_path, input_root, args.output)
            write_image(prediction, output_path)

        prediction_float = prediction.float().div(255.0).clamp(0.0, 1.0)
        gt_path = gt_index.get(key)
        target_float = None
        if gt_path is not None:
            target_float = read_image(gt_path, device).float().div(255.0).clamp(0.0, 1.0)
            if prediction_float.shape != target_float.shape:
                raise ValueError(
                    f"Prediction/GT shape mismatch for {key!r}: "
                    f"{tuple(prediction_float.shape)} != {tuple(target_float.shape)}"
                )

        # The CSV contains the image key, input path, prediction path, GT path, and metric values for each image.
        row = {
            "image": key,
            "input": str(input_path),
            "prediction": str(output_path) if output_path is not None else "",
            "gt": str(gt_path) if gt_path is not None else "",
        }
        for name, metric in metrics.items():
            mode = getattr(metric, "metric_mode", None)
            if mode == "FR":
                value = metric(prediction_float, target_float)
            elif mode == "NR":
                value = metric(prediction_float)
            else:
                raise ValueError(f"PyIQA metric {name!r} has unsupported mode {mode!r}")
            value = float(value.detach().mean().cpu()) if torch.is_tensor(value) else float(value)
            row[name] = value
            if math.isfinite(value):
                sums[name] += value
                counts[name] += 1
        rows.append(row)
        print(f"Evaluated {index}/{len(input_index)}: {key}", flush=True)

    summary_metrics = {
        name: (sums[name] / counts[name] if counts[name] else None)
        for name in metric_names
    }
    lower_is_better = sorted(
        name for name, metric in metrics.items() if getattr(metric, "lower_better", False)
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "gt": str(args.gt) if args.gt is not None else None,
        "output": str(args.output) if args.output is not None else None,
        "images": len(rows),
        "sampler": "jump",
        "nfe": args.nfe,
        "sigma": args.sigma,
        "metrics": summary_metrics,
        "finite_images": counts,
        "lower_is_better": lower_is_better,
        "higher_is_better": sorted(set(metric_names) - set(lower_is_better)),
    }

    # If no output directory is specified, the metrics are printed to stdout instead of being saved to disk.
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["image", "input", "prediction", "gt", *metrics],
            )
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
