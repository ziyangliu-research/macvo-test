#!/usr/bin/env python3
"""Seed benchmark RNGs and run the reproducible execution baseline.

For the formal serial baseline, ReSplat is pinned to CUDA's default stream.
The current ReSplat stack has operators that do not behave correctly when the
whole encoder is launched on a separate persistent CUDA stream: raw packet
self-render quality collapses despite explicit inter-stream waits. The validated
legacy ZipMap+ReSplat path and the default-stream diagnostic both produce the
expected packet quality.

Async scheduling remains available for future investigation, but this wrapper
only applies the stream override when ``--mode serial`` is requested.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch


def _requested_mode(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--mode" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--mode="):
            return token.split("=", 1)[1]
    return None


def _pin_serial_resplat_to_default_stream() -> None:
    """Make the formal serial baseline use the validated ReSplat stream path."""

    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_initialize = ResplatPacketGenerator.initialize
    if getattr(original_initialize, "_serial_default_stream_patch", False):
        return

    def initialize_on_default_stream(self: ResplatPacketGenerator) -> None:
        original_initialize(self)
        if self.device.type == "cuda":
            self.stream = torch.cuda.default_stream(self.device)

    initialize_on_default_stream._serial_default_stream_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.initialize = initialize_on_default_stream


def main() -> None:
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    mode = _requested_mode(sys.argv[1:])
    if mode == "serial":
        _pin_serial_resplat_to_default_stream()
        print(
            "[repro] serial baseline: ReSplat pinned to CUDA default stream",
            flush=True,
        )

    from run_pipeline_execution_benchmark import main as benchmark_main

    benchmark_main()


if __name__ == "__main__":
    main()
