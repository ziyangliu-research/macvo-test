#!/usr/bin/env python3
"""Reproducible serial benchmark with GraphDECO supervised in ReSplat's camera domain.

This diagnostic keeps MAC-VO on its native input frames, but replaces the backend
Observation with the left RGB image and pixel-space intrinsics obtained from the
same resize + center-crop transform used by ReSplat. It is intended to isolate
frontend/backend camera-domain mismatch without changing the normal benchmark.

For SH003 with the ReSplat experiment default image_shape=[320,320], both ReSplat
and GraphDECO therefore see the same 320x320 center-cropped camera domain.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

import run_pipeline_execution_benchmark as benchmark
from async_pipeline.backend_observation import make_resplat_domain_observation
from run_pipeline_execution_benchmark_repro import (
    _pin_serial_resplat_to_default_stream,
    _requested_mode,
)


class ResplatCameraSerialPipelineRunner(benchmark.SerialPipelineRunner):
    """Serial runner whose backend observations exactly match ReSplat preprocessing."""

    def __init__(self, config, pose_frontend, packet_generator, backend) -> None:
        super().__init__(config, pose_frontend, packet_generator, backend)
        original_iter_frames = self.pose_frontend.iter_frames

        def iter_frames_with_resplat_camera():
            for descriptor, frame, stereo_input, _raw_observation in original_iter_frames():
                observation = make_resplat_domain_observation(
                    self.packet_generator,
                    stereo_input,
                )
                height, width = observation.image.shape[-2:]
                self.pose_frontend.recorder.set_value(
                    descriptor.sequence_index,
                    "backend_observation_height",
                    int(height),
                )
                self.pose_frontend.recorder.set_value(
                    descriptor.sequence_index,
                    "backend_observation_width",
                    int(width),
                )
                self.pose_frontend.recorder.set_value(
                    descriptor.sequence_index,
                    "backend_observation_domain",
                    "resplat",
                )
                yield descriptor, frame, stereo_input, observation

        self.pose_frontend.iter_frames = iter_frames_with_resplat_camera


def main() -> None:
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    mode = _requested_mode(sys.argv[1:])
    if mode != "serial":
        raise ValueError(
            "run_pipeline_execution_benchmark_resplat_camera_repro.py "
            "currently supports --mode serial only"
        )

    _pin_serial_resplat_to_default_stream()
    benchmark.SerialPipelineRunner = ResplatCameraSerialPipelineRunner
    print(
        "[repro] serial baseline: ReSplat pinned to CUDA default stream; "
        "backend observation matched to ReSplat resize/crop camera domain",
        flush=True,
    )
    benchmark.main()


if __name__ == "__main__":
    main()
