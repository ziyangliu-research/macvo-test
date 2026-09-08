#!/usr/bin/env python3
"""Evaluate EuRoC ReSplat packets with the same evaluator used for P000/SH003.

This wrapper intentionally keeps ``evaluate_resplat_from_execution_baseline.py``
as the single scoring implementation and only installs the EuRoC-specific camera
contract before it is constructed:

- EuRoC synchronized/GT-covered stereo loader;
- sensor.yaml calibration + stereo rectification from the dataset loader;
- dynamic rectified K for ReSplat shared-tensor input;
- EuRoC full-FoV resize to the configured ReSplat network shape;
- optional temporal stride controlled by PIPELINE_FRAME_STRIDE.

It also pins ReSplat to CUDA's default stream, matching the validated P000/SH003
``evaluate_resplat_default_stream_repro.py`` path. No MAC-VO pose estimation,
packet fusion, GraphDECO optimization, pruning, replay, or global refinement is
run by the underlying evaluator.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def main() -> None:
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # Install EuRoC input/camera semantics first so the baseline evaluator builds
    # the exact same EuRoC frame path as the formal pipeline.
    from euroc_stride5_support import install_euroc_runtime_support

    install_euroc_runtime_support()

    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_initialize = ResplatPacketGenerator.initialize

    def initialize_on_default_stream(self: ResplatPacketGenerator) -> None:
        original_initialize(self)
        if self.device.type == "cuda":
            self.stream = torch.cuda.default_stream(self.device)

    ResplatPacketGenerator.initialize = initialize_on_default_stream
    print(
        "[diagnostic] EuRoC ReSplat-only evaluator: CUDA default stream; "
        f"frame_stride={os.environ.get('PIPELINE_FRAME_STRIDE', '5')}",
        flush=True,
    )

    from evaluate_resplat_from_execution_baseline import main as evaluator_main

    evaluator_main()


if __name__ == "__main__":
    main()
