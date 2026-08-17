#!/usr/bin/env python3
"""Run the formal TartanAir V1 comparison protocol on a serial pipeline.

This entry point is deliberately separate from the existing P000 benchmark code.
It keeps the validated serial MAC-VO -> ReSplat -> incremental 3DGS execution
path, but adds two comparison-specific controls without changing the legacy
train/test behavior:

- split.enabled=false: every input frame is inserted/optimized;
- evaluation.final_only=true: no intermediate render evaluation is performed;
  the final map is evaluated exactly once after all packets are processed.

The external TartanAir challenge GT pose file is also configurable through
``evaluation.gt_pose_file`` so V1 challenge sequences do not need a synthetic
``pose_lcam_front.txt`` inside the image directory.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_async_pipeline as base
from async_pipeline.contracts import FrameDescriptor
from async_pipeline.macvo_runtime import _pose7_xyzw_to_matrix, _tartan_from_cv
from async_pipeline.pipeline import AsyncPipelineConfig
from run_async_pipeline_metrics import (
    collect_predicted_trajectory,
    save_tum,
    trajectory_metric,
)
from run_pipeline_execution_benchmark import (
    SerialPipelineRunner,
    TimedBackend,
    TimedPacketGenerator,
    TimedPoseFrontend,
    TimingRecorder,
)


@dataclass(frozen=True)
class ComparisonPipelineConfig(AsyncPipelineConfig):
    """AsyncPipelineConfig with an explicit switch for the train/test split."""

    split_enabled: bool = True

    def is_test(self, descriptor: FrameDescriptor) -> bool:
        if not self.split_enabled:
            return False
        return super().is_test(descriptor)


class FinalOnlyEvaluationBackend:
    """Disable render metrics during updates and restore them for finalize()."""

    def __init__(self, wrapped: Any, *, final_only: bool) -> None:
        self._wrapped = wrapped
        self.config = wrapped.config
        self.final_only = bool(final_only)
        self.finalize_sec = 0.0
        self._process_config = (
            replace(self.config, evaluation_enabled=False)
            if self.final_only and self.config.evaluation_enabled
            else self.config
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def initialize(self) -> None:
        # Initialize W&B and the backend using the report configuration first,
        # then suppress only intermediate render evaluation during processing.
        self._wrapped.config = self.config
        self._wrapped.initialize()
        self._wrapped.config = self._process_config

    def process(self, update: Any) -> None:
        self._wrapped.process(update)

    def finalize(self) -> dict[str, Any]:
        # Restore evaluation_enabled so BackendEvaluationMixin.finalize() renders
        # the completed map exactly once.
        self._wrapped.config = self.config
        start = time.perf_counter()
        summary = self._wrapped.finalize()
        self.finalize_sec = time.perf_counter() - start
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "Config/Pipeline/"
            "MACVO_ReSplat_Serial_TartanAirV1_SH000_0_200_AllFrames.yaml"
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    parser.add_argument(
        "--skip_pose_metrics",
        action="store_true",
        help="Skip ATE calculation even if evaluation.gt_pose_file is configured.",
    )
    return parser.parse_args()


def resolve_comparison_paths(resolved: dict[str, Any]) -> None:
    evaluation = resolved.get("evaluation", {})
    gt_value = evaluation.get("gt_pose_file")
    if not gt_value:
        return
    macvo_repo = Path(resolved["paths"]["macvo_repo"])
    gt_path = base.absolute(gt_value, macvo_repo)
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground-truth pose file not found: {gt_path}")
    evaluation["gt_pose_file"] = str(gt_path)


def build_comparison_config(
    base_config: AsyncPipelineConfig,
    resolved: dict[str, Any],
    work_dir: Path,
) -> ComparisonPipelineConfig:
    split = resolved["split"]
    return ComparisonPipelineConfig(
        split_enabled=bool(split.get("enabled", True)),
        split_every=base_config.split_every,
        split_offset=base_config.split_offset,
        split_index_mode=base_config.split_index_mode,
        queue_size=base_config.queue_size,
        poll_timeout_sec=base_config.poll_timeout_sec,
        initialization_timeout_sec=base_config.initialization_timeout_sec,
        summary_path=work_dir / "comparison_pipeline_summary.json",
    )


def load_external_gt_trajectory(
    resolved: dict[str, Any],
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    evaluation = resolved.get("evaluation", {})
    pose_path_value = evaluation.get("gt_pose_file")
    if not pose_path_value:
        raise ValueError("evaluation.gt_pose_file is required for comparison ATE")
    pose_path = Path(pose_path_value)
    rows = np.loadtxt(pose_path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] != 7:
        raise ValueError(
            f"expected TartanAir pose rows tx ty tz qx qy qz qw, got {rows.shape}"
        )

    start = int(resolved["sequence"]["start_index"])
    end = int(resolved["sequence"]["end_index"])
    frame_indices = np.arange(start, min(end, start + count), dtype=np.int64)
    if frame_indices.size == 0 or int(frame_indices[-1]) >= rows.shape[0]:
        raise IndexError(
            f"requested GT rows [{start},{end}) outside pose file with {rows.shape[0]} rows"
        )

    tartan_from_cv = _tartan_from_cv(dtype=torch.float64)
    absolute: list[np.ndarray] = []
    for index in frame_indices:
        pose = _pose7_xyzw_to_matrix(torch.from_numpy(rows[int(index)]))
        absolute.append((pose @ tartan_from_cv).numpy())
    absolute_array = np.stack(absolute, axis=0)
    first_inverse = np.linalg.inv(absolute_array[0])
    relative = first_inverse[None] @ absolute_array
    return frame_indices, relative


def evaluate_external_pose(
    runner: SerialPipelineRunner,
    resolved: dict[str, Any],
    num_frames: int,
    output: Path,
) -> dict[str, Any]:
    predicted_all, valid_mask = collect_predicted_trajectory(runner, num_frames)
    frame_indices, gt_all = load_external_gt_trajectory(
        resolved, predicted_all.shape[0]
    )

    predicted = predicted_all[valid_mask]
    ground_truth = gt_all[valid_mask]
    valid_indices = frame_indices[valid_mask]
    if predicted.shape[0] < 3:
        raise RuntimeError("fewer than three valid MAC-VO poses are available")

    report = {
        "gt_pose_file": resolved["evaluation"]["gt_pose_file"],
        "gt_format": "tx ty tz qx qy qz qw (TartanAir NED)",
        "coordinate_convention": "metric OpenCV c2w, first frame identity",
        "num_requested_poses": int(predicted_all.shape[0]),
        "num_valid_poses": int(predicted.shape[0]),
        "num_skipped_need_interp": int((~valid_mask).sum()),
        "raw": trajectory_metric(predicted, ground_truth, alignment="raw"),
        "se3": trajectory_metric(predicted, ground_truth, alignment="se3"),
        "sim3": trajectory_metric(predicted, ground_truth, alignment="sim3"),
    }

    np.savez_compressed(
        output / "macvo_comparison_trajectory.npz",
        selected_original_indices=frame_indices,
        valid_mask=valid_mask,
        T_pred_c2w_opencv=predicted_all,
        T_gt_c2w_opencv=gt_all,
    )
    save_tum(output / "macvo_pred_valid.tum", valid_indices, predicted)
    save_tum(output / "macvo_gt_valid.tum", valid_indices, ground_truth)
    (output / "pose_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = base.absolute(args.config, root)
    config = base.load_yaml(config_path)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}")
        key, raw = item.split("=", 1)
        base.nested_set(config, key, base.normalize_override_value(key, raw))

    resolved = base.resolve(config, root)
    resolve_comparison_paths(resolved)
    base.validate_paths(resolved)

    work_dir = Path(resolved["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "resolved_comparison_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    template = base.build_system(resolved)
    comparison_config = build_comparison_config(
        template.config, resolved, work_dir
    )

    final_only = bool(resolved.get("evaluation", {}).get("final_only", False))
    backend_adapter = FinalOnlyEvaluationBackend(
        template.backend,
        final_only=final_only,
    )

    recorder = TimingRecorder()
    pose = TimedPoseFrontend(template.pose_frontend, recorder)
    packet = TimedPacketGenerator(template.packet_generator, recorder)
    backend = TimedBackend(backend_adapter, recorder)
    runner = SerialPipelineRunner(comparison_config, pose, packet, backend)

    summary = runner.run()
    rows = recorder.rows()
    frame_timing_path = work_dir / "frame_timing_log.json"
    frame_timing_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    wall = float(summary["streaming_wall_time_sec"])
    train_count = int(summary["num_train_frames"])
    frame_count = int(summary["num_frames"])
    finalize_sec = float(backend_adapter.finalize_sec)
    reconstruction_sec = max(0.0, wall - finalize_sec)

    comparison_summary: dict[str, Any] = {
        **summary,
        "execution_mode": "serial",
        "comparison_protocol": {
            "dataset_family": "TartanAir V1 Stereo Challenge",
            "frame_range": [
                int(resolved["sequence"]["start_index"]),
                int(resolved["sequence"]["end_index"]) - 1,
            ],
            "split_enabled": bool(comparison_config.split_enabled),
            "all_frames_inserted": not bool(comparison_config.split_enabled),
            "final_evaluation_only": final_only,
            "render_metric_scope": (
                "all processed/seen views"
                if not comparison_config.split_enabled
                else "configured train/test split"
            ),
        },
        "frame_timing_log": str(frame_timing_path),
        "input_fps_excluding_initialization": (
            frame_count / wall if wall > 0 else 0.0
        ),
        "map_update_fps_excluding_initialization": (
            train_count / wall if wall > 0 else 0.0
        ),
        "finalize_and_final_evaluation_sec": finalize_sec,
        "reconstruction_wall_time_before_finalize_sec": reconstruction_sec,
    }

    backend_summary = comparison_summary.get("backend") or {}
    final_metrics = backend_summary.get("final_metrics") or {}
    if comparison_config.split_enabled:
        comparison_summary["comparison_render_metrics"] = {
            "train_inserted": final_metrics.get("train_inserted", {}),
            "test_all": final_metrics.get("test_all", {}),
        }
    else:
        # With split disabled every camera is an inserted/optimized camera. Rename
        # the existing train_inserted result at the report layer so comparison
        # tables do not incorrectly call it a held-out train/test metric.
        comparison_summary["comparison_render_metrics"] = {
            "all_processed": final_metrics.get("train_inserted", {}),
        }

    if not args.skip_pose_metrics and resolved.get("evaluation", {}).get(
        "gt_pose_file"
    ):
        comparison_summary["pose_metrics"] = evaluate_external_pose(
            runner,
            resolved,
            frame_count,
            work_dir,
        )

    output = work_dir / "comparison_benchmark_summary.json"
    output.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    print("\n[TartanAir V1 comparison summary]", flush=True)
    print(json.dumps(comparison_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
