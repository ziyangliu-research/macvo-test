#!/usr/bin/env python3
"""Run the ReSplat-only evaluator with ReSplat forced onto the current CUDA stream.

This is a narrow diagnostic wrapper. It keeps the pipeline/config/evaluator unchanged
and changes only the CUDA stream used by ``ResplatPacketGenerator`` after model
initialization. The wrapper exists because ReSplat currently produces correct P000
packets on the default/current stream but degraded packets on its dedicated stream,
even after explicit producer->consumer stream synchronization.
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

    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_initialize = ResplatPacketGenerator.initialize

    def initialize_on_current_stream(self: ResplatPacketGenerator) -> None:
        original_initialize(self)
        # Critical diagnostic change: execute ReSplat on the caller/default stream.
        # This matches the validated legacy ZipMap+ReSplat execution path and avoids
        # non-default-stream behavior in ReSplat/custom CUDA operators.
        self.stream = torch.cuda.current_stream(self.device)

    ResplatPacketGenerator.initialize = initialize_on_current_stream

    from evaluate_resplat_from_execution_baseline import main as evaluator_main

    evaluator_main()


if __name__ == "__main__":
    main()
