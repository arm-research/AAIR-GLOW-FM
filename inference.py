# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Enhance images with Glow-FM."""

import argparse
from pathlib import Path

from utils.image_enhancement import (
    enhance,
    input_images,
    load_pipeline,
    prediction_path,
    read_image,
    resolve_device,
    write_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, help="Optional prediction output directory")
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

    # The checkpoint contains both the architecture configuration and weights.
    device = resolve_device(args.device)
    network, encoder = load_pipeline(args.checkpoint, device)

    input_root = args.input if args.input.is_dir() else args.input.parent
    inputs = input_images(args.input, exclude=args.output)
    for input_path in inputs:
        # Images are loaded directly on the requested inference device.
        image = read_image(input_path, device)
        prediction = enhance(
            network,
            encoder,
            image,
            nfe=args.nfe,
            sigma=args.sigma,
        )

        # If no output directory is specified, the enhanced image is not saved to disk.
        if args.output is None:
            print(f"Enhanced {input_path}", flush=True)
            continue
        output_path = prediction_path(input_path, input_root, args.output)
        write_image(prediction, output_path)
        print(f"Enhanced {input_path} -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
