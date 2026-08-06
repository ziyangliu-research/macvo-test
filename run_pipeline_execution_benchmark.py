#!/usr/bin/env python3
"""Benchmark matched serial and asynchronous executions of the same pipeline.

The benchmark deliberately reuses exactly the same persistent MAC-VO, ReSplat,
and incremental GraphDECO classes. The only experimental variable is scheduling:

- async: bounded worker queues and ordered frontend/backend joining;
- serial: the same stages are executed synchronously with no stage overlap.

Both modes write frame-level timing records suitable for cumulative completion,
per-update interval, and nominal sensor-to-map latency plots.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import run_async_pipeline as base
from async_pipeline.contracts import BackendUpdate, FrameDescriptor, PoseEstimate
from async_pipeline.joiner import OrderedFrontendJoiner
from async_pipeline.pipeline import AsyncPipelineRunner


class TimingRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._origin: float | None = None
        self._first_timestamp_ns: int | None = None
        self._records: dict[int, dict[str, Any]] = {}

    def _relative(self, absolute: float) -> float:
        if self._origin is None:
            self._origin = absolute
        return absolute - self._origin

    def register(self, descriptor: FrameDescriptor) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._first_timestamp_ns is None:
                self._first_timestamp_ns = int(descriptor.timestamp_ns)
            record = self._records.setdefault(descriptor.sequence_index, {})
            record.update(
                {
                    "sequence_index": int(descriptor.sequence_index),
                    "frame_index": int(descriptor.frame_index),
                    "timestamp_ns": int(descriptor.timestamp_ns),
                    "is_test": bool(descriptor.is_test),
                    "input_ready_sec": self._relative(now),
                }
            )

    def update_descriptor(self, descriptor: FrameDescriptor) -> None:
        with self._lock:
            record = self._records.setdefault(descriptor.sequence_index, {})
            record.update(
                {
                    "sequence_index": int(descriptor.sequence_index),
                    "frame_index": int(descriptor.frame_index),
                    "timestamp_ns": int(descriptor.timestamp_ns),
                    "is_test": bool(descriptor.is_test),
                }
            )

    def mark(self, sequence_index: int, key: str, absolute: float | None = None) -> None:
        when = time.perf_counter() if absolute is None else absolute
        with self._lock:
            record = self._records.setdefault(sequence_index, {})
            record[key] = self._relative(when)

    def set_value(self, sequence_index: int, key: str, value: Any) -> None:
        with self._lock:
            record = self._records.setdefault(sequence_index, {})
            record[key] = value

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            raw = [dict(self._records[key]) for key in sorted(self._records)]
            first_timestamp_ns = self._first_timestamp_ns

        previous_train_completion: float | None = None
        previous_any_completion: float | None = None
        rows: list[dict[str, Any]] = []
        for record in raw:
            timestamp_ns = int(record.get("timestamp_ns", 0))
            nominal_arrival = 0.0
            if first_timestamp_ns is not None:
                nominal_arrival = (timestamp_ns - first_timestamp_ns) / 1e9
            record["nominal_sensor_arrival_sec"] = nominal_arrival

            for prefix in ("pose_task", "packet", "backend"):
                start_key = f"{prefix}_start_sec"
                end_key = f"{prefix}_end_sec"
                if start_key in record and end_key in record:
                    record[f"{prefix}_duration_sec"] = max(
                        0.0, float(record[end_key]) - float(record[start_key])
                    )

            if "backend_end_sec" in record:
                completion = float(record["backend_end_sec"])
                record["map_completion_elapsed_sec"] = completion
                record["nominal_sensor_to_map_latency_sec"] = max(
                    0.0, completion - nominal_arrival
                )
                if "input_ready_sec" in record:
                    record["loader_ready_to_map_latency_sec"] = max(
                        0.0, completion - float(record["input_ready_sec"])
                    )
                if previous_any_completion is not None:
                    record["completion_interval_all_updates_sec"] = max(
                        0.0, completion - previous_any_completion
                    )
                previous_any_completion = completion
                if not bool(record.get("is_test", False)):
                    if previous_train_completion is not None:
                        record["completion_interval_train_updates_sec"] = max(
                            0.0, completion - previous_train_completion
                        )
                    previous_train_completion = completion
            rows.append(record)
        return rows


class TimedPoseFrontend:
    def __init__(self, wrapped: Any, recorder: TimingRecorder) -> None:
        self._wrapped = wrapped
        self.recorder = recorder
        self.config = wrapped.config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def initialize(self) -> None:
        self._wrapped.initialize()

    def iter_frames(self):
        for descriptor, frame, stereo_input, observation in self._wrapped.iter_frames():
            self.recorder.register(descriptor)
            yield descriptor, frame, stereo_input, observation

    def process(self, descriptor: FrameDescriptor, frame: Any) -> list[PoseEstimate]:
        start = time.perf_counter()
        self.recorder.mark(descriptor.sequence_index, "pose_task_start_sec", start)
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


class TimedPacketGenerator:
    def __init__(self, wrapped: Any, recorder: TimingRecorder) -> None:
        self._wrapped = wrapped
        self.recorder = recorder
        self.config = wrapped.config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def initialize(self) -> None:
        self._wrapped.initialize()

    def infer(self, frame_input, **kwargs):
        sequence_index = frame_input.descriptor.sequence_index
        start = time.perf_counter()
        self.recorder.mark(sequence_index, "packet_start_sec", start)
        result = self._wrapped.infer(frame_input, **kwargs)
        end = time.perf_counter()
        self.recorder.mark(sequence_index, "packet_end_sec", end)
        self.recorder.set_value(
            sequence_index, "num_packet_gaussians", result.packet.num_gaussians
        )
        return result

    def close(self) -> None:
        self._wrapped.close()


class TimedBackend:
    def __init__(self, wrapped: Any, recorder: TimingRecorder) -> None:
        self._wrapped = wrapped
        self.recorder = recorder
        self.config = getattr(wrapped, "config", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def initialize(self) -> None:
        self._wrapped.initialize()

    def process(self, update: BackendUpdate) -> None:
        sequence_index = update.descriptor.sequence_index
        self.recorder.update_descriptor(update.descriptor)
        self.recorder.set_value(sequence_index, "join_wait_sec", update.join_wait_sec)
        start = time.perf_counter()
        self.recorder.mark(sequence_index, "backend_start_sec", start)
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

    def finalize(self) -> dict[str, Any]:
        return self._wrapped.finalize()


class SerialPipelineRunner:
    """Matched no-overlap baseline using the same persistent component classes."""

    def __init__(
        self,
        config,
        pose_frontend: TimedPoseFrontend,
        packet_generator: TimedPacketGenerator,
        backend: TimedBackend,
    ) -> None:
        self.config = config
        self.pose_frontend = pose_frontend
        self.packet_generator = packet_generator
        self.backend = backend

    def _process_updates(self, updates: Iterable[BackendUpdate]) -> int:
        count = 0
        for update in updates:
            self.backend.process(update)
            count += 1
        return count

    def run(self) -> dict[str, Any]:
        init_start = time.perf_counter()
        self.pose_frontend.initialize()
        self.packet_generator.initialize()
        self.backend.initialize()
        initialization_sec = time.perf_counter() - init_start

        joiner = OrderedFrontendJoiner()
        wall_start = time.perf_counter()
        num_frames = submitted_train = submitted_test = emitted_updates = 0

        for descriptor, frame, stereo_input, observation in self.pose_frontend.iter_frames():
            descriptor = replace(
                descriptor,
                is_test=self.config.is_test(descriptor),
            )
            self.pose_frontend.recorder.update_descriptor(descriptor)
            stereo_input.descriptor = descriptor
            observation.descriptor = descriptor
            joiner.register(descriptor)
            num_frames += 1
            emitted_updates += self._process_updates(joiner.add_observation(observation))

            # Pose first: a newly arriving timestamp commits the previous pose.
            # Any now-ready map update is completed before the next stage starts.
            for estimate in self.pose_frontend.process(descriptor, frame):
                emitted_updates += self._process_updates(joiner.add_pose(estimate))

            if descriptor.is_test:
                submitted_test += 1
            else:
                submitted_train += 1
                packet = self.packet_generator.infer(
                    stereo_input,
                    output_frame="left_camera_local",
                ).packet
                emitted_updates += self._process_updates(joiner.add_packet(packet))

        joiner.close_registration()
        for estimate in self.pose_frontend.terminate():
            emitted_updates += self._process_updates(joiner.add_pose(estimate))
        if not joiner.finished:
            raise RuntimeError("serial joiner did not emit all registered frames")

        self.packet_generator.close()
        backend_summary = self.backend.finalize()
        streaming_wall_time_sec = time.perf_counter() - wall_start
        summary = {
            "pipeline": "serial no-overlap class-based MAC-VO + local ReSplat + incremental backend",
            "execution_mode": "serial",
            "num_frames": num_frames,
            "num_train_frames": submitted_train,
            "num_test_frames": submitted_test,
            "num_backend_updates": emitted_updates,
            "strict_split": {
                "split_every": self.config.split_every,
                "split_offset": self.config.split_offset,
                "split_index_mode": self.config.split_index_mode,
                "test_packets_generated": False,
            },
            "runtime_contract": {
                "model_initialization": "once per persistent component",
                "frontend_io": "one decoded stereo frame shared by adapters",
                "packet_handoff": self.packet_generator.config.handoff_mode,
                "backend_order": "strict sequence order",
                "stage_overlap": False,
            },
            "initialization_sec": initialization_sec,
            "streaming_wall_time_sec": streaming_wall_time_sec,
            "backend": backend_summary,
        }
        output = self.config.summary_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["serial", "async"], required=True)
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
        help="Also compute Raw/SE3/Sim3 ATE after the run.",
    )
    return parser.parse_args()


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
    base.validate_paths(resolved)
    work_dir = Path(resolved["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "resolved_execution_benchmark_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    template = base.build_system(resolved)
    recorder = TimingRecorder()
    pose = TimedPoseFrontend(template.pose_frontend, recorder)
    packet = TimedPacketGenerator(template.packet_generator, recorder)
    backend = TimedBackend(template.backend, recorder)

    if args.mode == "async":
        runner = AsyncPipelineRunner(template.config, pose, packet, backend)
    else:
        runner = SerialPipelineRunner(template.config, pose, packet, backend)

    summary = runner.run()
    summary["execution_mode"] = args.mode
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
    print("\n[execution benchmark summary]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
