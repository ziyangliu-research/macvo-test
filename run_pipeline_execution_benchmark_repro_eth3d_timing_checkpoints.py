#!/usr/bin/env python3
"""ETH3D full benchmark timing wrapper for 10/15/20/25/30-pass refinement."""
from __future__ import annotations

from eth3d_rectified_support import install_eth3d_runtime_support

install_eth3d_runtime_support()

import run_pipeline_execution_benchmark_repro_posthoc_global_refine_timing_checkpoints as timing


if __name__ == "__main__":
    timing._disable_all_evaluation()
    timing._install_timing_refinement()
    timing.safe.repro.main()
