#!/usr/bin/env python3
"""Seed all benchmark RNGs before invoking the execution benchmark runner."""
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

    from run_pipeline_execution_benchmark import main as benchmark_main

    benchmark_main()


if __name__ == "__main__":
    main()
