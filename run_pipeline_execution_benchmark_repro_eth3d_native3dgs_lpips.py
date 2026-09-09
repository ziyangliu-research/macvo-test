#!/usr/bin/env python3
"""ETH3D wrapper for native-GraphDECO post-hoc global refinement."""
from __future__ import annotations

import run_async_pipeline_metrics as pose_metrics
from eth3d_rectified_support import evaluate_pose_eth3d, install_eth3d_runtime_support

# Install ETH3D rectified-camera/runtime semantics before pipeline construction.
install_eth3d_runtime_support()


def _evaluate_pose_from_runner(runner, resolved, num_frames, output):
    return evaluate_pose_eth3d(runner.pose_frontend, resolved, num_frames, output)


pose_metrics.evaluate_pose = _evaluate_pose_from_runner

import run_pipeline_execution_benchmark_repro_posthoc_global_refine_native3dgs_lpips as native


if __name__ == "__main__":
    native.quality_helpers._disable_intermediate_online_evaluation()
    native.quality_helpers._install_lpips_evaluator()
    native._install_native_graphdeco_refinement()
    native.safe.repro.main()
