#!/usr/bin/env python3
"""ReSplat-only evaluator for rectified ETH3D using the validated default stream."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from eth3d_rectified_support import install_eth3d_runtime_support


def main() -> None:
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    install_eth3d_runtime_support()

    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_initialize = ResplatPacketGenerator.initialize

    def initialize_on_default_stream(self: ResplatPacketGenerator) -> None:
        original_initialize(self)
        if self.device.type == "cuda":
            # Mandatory: ReSplat/custom CUDA operators are known to produce
            # incorrect packets on a persistent non-default stream.
            self.stream = torch.cuda.default_stream(self.device)

    ResplatPacketGenerator.initialize = initialize_on_default_stream
    print("[diagnostic] ETH3D ReSplat pinned to CUDA default stream", flush=True)

    from evaluate_resplat_from_execution_baseline import main as evaluator_main
    evaluator_main()


if __name__ == "__main__":
    main()
