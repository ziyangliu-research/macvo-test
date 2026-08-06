#!/usr/bin/env python3
"""Benchmark asynchronous queues with mutually exclusive single-GPU execution.

This is the Experiment-C diagnostic mode:

- the MAC-VO, ReSplat, and incremental-backend workers remain asynchronous;
- persistent models, bounded queues, strict split, and ordered joining are kept;
- one process-wide GPU gate serializes each module-level GPU-heavy call;
- CUDA is synchronized before the gate is released so kernels cannot overlap
  after the Python critical section ends.

The implementation is intentionally coarse grained. Its purpose is to test
whether the negative speedup of naive single-GPU overlap is mainly caused by
GPU resource contention. It is not presented as the final scheduling policy.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import run_async_pipeline as base
from async_pipeline.contracts import BackendUpdate, FrameDescriptor, PoseEstimate
from async_pipeline.pipeline import AsyncPipelineRunner
from run_pipeline_execution_benchmark import (
    TimedBackend,
    TimedPacketGenerator,
    TimedPoseFrontend,
    TimingRecorder,
)


class GpuExecutionGate:
    """Serialize module-level GPU work and report lock wait/hold statistics."""

    def __init__(self, recorder: TimingRecorder) -> None:
        self._lock = threading.Lock()
        self._recorder = recorder
        self._stats: dict[str, dict[str, float | int]] = {}

    @contextmanager
    def section(self, stage: str, sequence_index: int | None) -> Iterator[None]:
        request_time = time.perf_counter()
        self._lock.acquire()
        acquired_time = time.perf_counter()
        wait_sec = acquired_time - request_time

        if sequence_index is not None:
            self._recorder.set_value(
                sequence_index, f"{stage}_gpu_gate_wait_sec", wait_sec
            )
            self._recorder.mark(
                sequence_index, f"{stage}_gpu_gate_acquired_sec", acquired_time
            )

        try:
            yield
            # A Python lock only serializes host submission. Synchronizing here
            # ensures that kernels submitted by this section have completed
            # before another worker is allowed to submit its GPU workload.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            released_time = time.perf_counter()
            held_sec = released_time - acquired_time
            if sequence_index is not None:
                self._recorder.set_value(
                    sequence_index, f"{stage}_gpu_gate_held_sec", held_sec
                )
                self._recorder.mark(
                    sequence_index, f"{stage}_gpu_gate_released_sec", released_time
                )

            stats = self._stats.setdefault(
                stage,
                {
                    "count": 0,
                    "total_wait_sec": 0.0,
                    "max_wait_sec": 0.0,
                    "total_held_sec": 0.0,
                    "max_held_sec": 0.0,
                },
            )
            stats["count"] = int(stats["count"]) + 1
            stats["total_wait_sec"] = float(stats["total_wait_sec"]) + wait_sec
            stats["max_wait_sec"] = max(float(stats["max_wait_sec"]), wait_sec)
            stats["total_held_sec"] = float(stats["total_held_sec"]) + held_sec
            stats["max_held_sec"] = max(float(stats["max_held_sec"]), held_sec)
            self._lock.release()

    def summary(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for stage, raw in self._stats.items():
            count = int(raw["count"])
            output[stage] = {
                **raw,
                "mean_wait_sec": (
                    0.0 if count == 0 else float(raw["total_wait_sec"]) / count
                ),
                "mean_held_sec": (
                    0.0 if count == 0 else float(raw["total_held_sec"]) / count
                ),
            }
        return output


class ResourceAwarePoseFrontend(TimedPoseFrontend):
    def __init__(
        self,
        wrapped: Any,
        recorder: TimingRecorder,
        gate: GpuExecutionGate,
    ) -> None:
        super().__init__(wrapped, recorder)
        self.gate = gate

    def process(self, descriptor: FrameDescriptor, frame: Any) -> list[PoseEstimate]:
        start = time.perf_counter()
        self.recorder.mark(descriptor.sequence_index, "pose_task_start_sec", start)
        with self.gate.section("pose", descriptor.sequence_index):
            output = self._wrapped.process(descriptor, frame)
        end = time.perf_counter()
        self.recorder.mark(descriptor.sequence_index, "pose_task_end_sec", end)
        for estimate in output:
            self.recorder.mark(
                estimate.descriptor.sequence_index, "pose_commit_ready_sec", end
            )
        return output

    def terminate(self) -> list[PoseEstimate]:
        start = time.perf_counter()
        with self.gate.section("pose_terminate", None):
            output = self._wrapped.terminate()
        end = time.perf_counter()
        for estimate in output:
            self.recorder.mark(
                estimate.descriptor.sequence_index, "pose_commit_ready_sec", end
            )
            self.recorder.set_value(
                estimate.descriptor.sequence_index,
                "pose_flushed_at_termination",
                True,
            )
        if output:
            self.recorder.set_value(
                output[-1].descriptor.sequence_index,
                "pose_termination_duration_sec",
                end - start,
            )
        return output


class ResourceAwarePacketGenerator(TimedPacketGenerator):
    def __init__(
        self,
        wrapped: Any,
        recorder: TimingRecorder,
        gate: GpuExecutionGate,
    ) -> None:
        super().__init__(wrapped, recorder)
        self.gate = gate

    def infer(self, frame_input, **kwargs):
        sequence_index = frame_input.descriptor.sequence_index
        start = time.perf_counter()
        self.recorder.mark(sequence_index, "packet_start_sec", start)
        with self.gate.section("packet", sequence_index):
            result = self._wrapped.infer(frame_input, **kwargs)
        end = time.perf_counter()
        self.recorder.mark(sequence_index, "packet_end_sec", end)
        self.recorder.set_value(
            sequence_index, "num_packet_gaussians", result.packet.num_gaussians
        )
        return result


class ResourceAwareBackend(TimedBackend):
    def __init__(
        self,
        wrapped: Any,
        recorder: TimingRecorder,
        gate: GpuExecutionGate,
    ) -> None:
        super().__init__(wrapped, recorder)
        self.gate = gate

    def process(self, update: BackendUpdate) -> None:
        sequence_index = update.descriptor.sequence_index
        self.recorder.update_descriptor(update.descriptor)
        self.recorder.set_value(sequence_index, "join_wait_sec", update.join_wait_sec)
        start = time.perf_counter()
        self.recorder.mark(sequence_index, "backend_start_sec", start)
        with self.gate.section("backend", sequence_index):
            self._wrapped.process(update)
        end = time.perf_counter()
        self.recorder.mark(sequence_index, "backend_end_sec", end)
        if getattr(self._wrapped, "gaussians", None) is not None:
            self.recorder.set_value(
                sequence_index,
                "num_map_gaussians",
                int(self._wrapped.gaussians.get_xyz.shape[0]),
            )
        self.recorder.set_value(
            sequence_index,
            "global_iteration",
            int(getattr(self._wrapped, "global_iteration", 0)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_Async_Fast_P000_0_50.yaml",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    parser.add_argument(
        "--with_pose_metrics",
        action="store_true",
        help="Also compute Raw/SE3/Sim3 ATE after the timed run.",
    )
    return parser.parse_args()


def seed_benchmark() -> int:
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed


def main() -> None:
    args = parse_args()
    seed = seed_benchmark()
    root = Path(__file__).resolve().parent
    config_path = base.absolute(args.config, root)
    config = base.load_yaml(config_path)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}")
        key, raw = item.split("=", 1)
        base.nested_set(config, key, base.normalize_override_value(key, raw))

    resolved = base.resolve(config, root)
    base.validate_paths(resolved)
    work_dir = Path(resolved["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "resolved_execution_benchmark_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    template = base.build_system(resolved)
    recorder = TimingRecorder()
    gate = GpuExecutionGate(recorder)
    pose = ResourceAwarePoseFrontend(template.pose_frontend, recorder, gate)
    packet = ResourceAwarePacketGenerator(template.packet_generator, recorder, gate)
    backend = ResourceAwareBackend(template.backend, recorder, gate)
    runner = AsyncPipelineRunner(template.config, pose, packet, backend)

    summary = runner.run()
    summary["execution_mode"] = "resource_aware_async"
    summary["benchmark_seed"] = seed
    summary["gpu_scheduling"] = {
        "policy": "coarse_grained_mutual_exclusion",
        "scope": "complete module process/infer/update call",
        "cuda_synchronize_before_release": True,
        "worker_threads_and_queues_remain_async": True,
    }
    summary["gpu_gate"] = gate.summary()
    summary["frame_timing_log"] = str(work_dir / "frame_timing_log.json")
    summary["streaming_fps_excluding_initialization"] = (
        float(summary["num_frames"]) / float(summary["streaming_wall_time_sec"])
    )
    summary["end_to_end_sec_including_initialization"] = (
        float(summary["initialization_sec"])
        + float(summary["streaming_wall_time_sec"])
    )

    rows = recorder.rows()
    (work_dir / "frame_timing_log.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    if args.with_pose_metrics:
        from run_async_pipeline_metrics import evaluate_pose

        summary["pose_metrics"] = evaluate_pose(
            runner,
            resolved,
            int(summary["num_frames"]),
            work_dir,
        )

    benchmark_output = work_dir / "execution_benchmark_summary.json"
    benchmark_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[resource-aware async benchmark summary]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
