#!/usr/bin/env python3
"""ETH3D full benchmark quality wrapper with 10/15/20/25/30-pass checkpoints."""
from __future__ import annotations

import run_async_pipeline_metrics as pose_metrics
from eth3d_rectified_support import evaluate_pose_eth3d, install_eth3d_runtime_support

# Dataset/runtime semantics must be installed before the benchmark constructs the
# frontend / ReSplat adapter.
install_eth3d_runtime_support()


def _evaluate_pose_from_runner(runner, resolved, num_frames, output):
    return evaluate_pose_eth3d(runner.pose_frontend, resolved, num_frames, output)


pose_metrics.evaluate_pose = _evaluate_pose_from_runner

import run_pipeline_execution_benchmark_repro_posthoc_global_refine_checkpoints_lpips as quality


if __name__ == "__main__":
    quality._disable_intermediate_online_evaluation()
    quality._install_lpips_evaluator()
    quality._install_checkpoint_refinement()
    quality.safe.repro.main()
