#!/usr/bin/env python3
"""Seed RNGs before running the TartanAir V1 comparison protocol."""
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

    from run_tartanair_v1_comparison import main as comparison_main

    comparison_main()


if __name__ == "__main__":
    main()
