#!/usr/bin/env python3
"""Run the current staged MAC-VO + ReSplat + incremental 3DGS pipeline.

This runner matches the execution policy selected after profiling the single-GPU
system:

1. complete the frontend stage for timestamp t;
2. only after the frontend stage is idle, run any now-ready backend update;
3. never overlap the incremental 3DGS backend with either frontend module.

Two frontend schedules are supported:

- serial: MAC-VO then ReSplat in the coordinator thread;
- parallel: MAC-VO and ReSplat run concurrently in two persistent worker threads,
  followed by a hard barrier before backend execution.

The existing fully asynchronous runner is intentionally left unchanged for
future experiments.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import run_async_pipeline as base
from async_pipeline.contracts import BackendUpdate
from async_pipeline.joiner import OrderedFrontendJoiner
from run_pipeline_execution_benchmark import (
    TimedBackend,
    TimedPacketGenerator,
    TimedPoseFrontend,
    TimingRecorder,
)


class StagedPipelineRunner:
    """Frontend-first pipeline with no frontend/backend overlap."""

    def __init__(
        self,
        config: Any,
        pose_frontend: TimedPoseFrontend,
        packet_generator: TimedPacketGenerator,
        backend: TimedBackend,
        *,
        frontend_mode: str,
    ) -> None:
        if frontend_mode not in {"serial", "parallel"}:
            raise ValueError(f"unknown frontend_mode {frontend_mode}")
        self.config = config
        self.pose_frontend = pose_frontend
        self.packet_generator = packet_generator
        self.backend = backend
        self.frontend_mode = frontend_mode

    def _process_updates(self, updates: Iterable[BackendUpdate]) -> int:
        count = 0
        for update in updates:
            # This is deliberately synchronous. At this point all frontend work
            # for the current timestamp has completed.
            self.backend.process(update)
            count += 1
        return count

    def run(self) -> dict[str, Any]:
        pose_executor: ThreadPoolExecutor | None = None
        packet_executor: ThreadPoolExecutor | None = None

        init_start = time.perf_counter()
        if self.frontend_mode == "parallel":
            # One persistent thread owns each frontend. Initialize sequentially
            # to avoid Hydra/CUDA allocator setup races while preserving ownership
            # for all later calls.
            pose_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pose-frontend"
            )
            packet_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="resplat-frontend"
            )
            pose_executor.submit(self.pose_frontend.initialize).result()
            packet_executor.submit(self.packet_generator.initialize).result()
        else:
            self.pose_frontend.initialize()
            self.packet_generator.initialize()

        # Backend always belongs to the coordinator thread in this runner.
        self.backend.initialize()
        initialization_sec = time.perf_counter() - init_start

        joiner = OrderedFrontendJoiner()
        wall_start = time.perf_counter()
        num_frames = submitted_train = submitted_test = emitted_updates = 0

        try:
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

                # Observation registration itself should normally not release an
                # update, but keep the returned list for contract completeness.
                ready_updates = list(joiner.add_observation(observation))

                frontend_start = time.perf_counter()
                self.pose_frontend.recorder.mark(
                    descriptor.sequence_index,
                    "frontend_stage_start_sec",
                    frontend_start,
                )

                packet_result = None
                if self.frontend_mode == "parallel":
                    assert pose_executor is not None
                    assert packet_executor is not None
                    pose_future = pose_executor.submit(
                        self.pose_frontend.process,
                        descriptor,
                        frame,
                    )
                    packet_future = None
                    if descriptor.is_test:
                        submitted_test += 1
                    else:
                        submitted_train += 1
                        packet_future = packet_executor.submit(
                            self.packet_generator.infer,
                            stereo_input,
                            output_frame="left_camera_local",
                        )

                    # Barrier: backend cannot start until every frontend task that
                    # exists for this timestamp has completed.
                    pose_estimates = pose_future.result()
                    if packet_future is not None:
                        packet_result = packet_future.result()
                else:
                    pose_estimates = self.pose_frontend.process(descriptor, frame)
                    if descriptor.is_test:
                        submitted_test += 1
                    else:
                        submitted_train += 1
                        packet_result = self.packet_generator.infer(
                            stereo_input,
                            output_frame="left_camera_local",
                        )

                frontend_end = time.perf_counter()
                self.pose_frontend.recorder.mark(
                    descriptor.sequence_index,
                    "frontend_stage_end_sec",
                    frontend_end,
                )
                self.pose_frontend.recorder.set_value(
                    descriptor.sequence_index,
                    "frontend_stage_duration_sec",
                    max(0.0, frontend_end - frontend_start),
                )
                self.pose_frontend.recorder.set_value(
                    descriptor.sequence_index,
                    "frontend_mode",
                    self.frontend_mode,
                )

                # Add all frontend products only after the barrier. Packet first is
                # useful with one-frame-delayed MAC-VO: when pose(t-1) is committed,
                # packet(t) has already finished and no frontend GPU work remains.
                if packet_result is not None:
                    ready_updates.extend(joiner.add_packet(packet_result.packet))
                for estimate in pose_estimates:
                    ready_updates.extend(joiner.add_pose(estimate))

                # Hard frontend/backend boundary.
                emitted_updates += self._process_updates(ready_updates)

            joiner.close_registration()

            # Preserve pose-thread ownership in parallel mode during MAC-VO flush.
            if self.frontend_mode == "parallel":
                assert pose_executor is not None
                final_poses = pose_executor.submit(self.pose_frontend.terminate).result()
            else:
                final_poses = self.pose_frontend.terminate()

            final_updates: list[BackendUpdate] = []
            for estimate in final_poses:
                final_updates.extend(joiner.add_pose(estimate))
            emitted_updates += self._process_updates(final_updates)

            if not joiner.finished:
                raise RuntimeError("staged joiner did not emit all registered frames")

            if self.frontend_mode == "parallel":
                assert packet_executor is not None
                packet_executor.submit(self.packet_generator.close).result()
            else:
                self.packet_generator.close()

            backend_summary = self.backend.finalize()
        finally:
            if pose_executor is not None:
                pose_executor.shutdown(wait=True, cancel_futures=False)
            if packet_executor is not None:
                packet_executor.shutdown(wait=True, cancel_futures=False)

        streaming_wall_time_sec = time.perf_counter() - wall_start
        map_update_fps = (
            submitted_train / streaming_wall_time_sec
            if streaming_wall_time_sec > 0
            else 0.0
        )
        input_fps = (
            num_frames / streaming_wall_time_sec
            if streaming_wall_time_sec > 0
            else 0.0
        )

        summary = {
            "pipeline": (
                "staged MAC-VO + ReSplat frontend followed by synchronous "
                "incremental backend"
            ),
            "execution_mode": f"staged_frontend_{self.frontend_mode}",
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
                "frontend_mode": self.frontend_mode,
                "frontend_barrier_before_backend": True,
                "frontend_backend_overlap": False,
                "backend_thread": "coordinator",
                "packet_handoff": self.packet_generator.config.handoff_mode,
                "backend_order": "strict sequence order",
            },
            "initialization_sec": initialization_sec,
            "streaming_wall_time_sec": streaming_wall_time_sec,
            "input_fps_excluding_initialization": input_fps,
            "map_update_fps_excluding_initialization": map_update_fps,
            "sec_per_train_packet": (
                streaming_wall_time_sec / submitted_train
                if submitted_train > 0
                else None
            ),
            "backend": backend_summary,
        }
        output = self.config.summary_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontend_mode",
        choices=["serial", "parallel"],
        default="serial",
        help=(
            "serial: MAC-VO then ReSplat; parallel: MAC-VO || ReSplat. "
            "Backend never overlaps either frontend."
        ),
    )
    parser.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_Serial_Report_P000_0_50.yaml",
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
    (work_dir / "resolved_staged_pipeline_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    template = base.build_system(resolved)
    recorder = TimingRecorder()
    pose = TimedPoseFrontend(template.pose_frontend, recorder)
    packet = TimedPacketGenerator(template.packet_generator, recorder)
    backend = TimedBackend(template.backend, recorder)

    runner = StagedPipelineRunner(
        template.config,
        pose,
        packet,
        backend,
        frontend_mode=args.frontend_mode,
    )
    summary = runner.run()

    rows = recorder.rows()
    timing_path = work_dir / "frame_timing_log.json"
    timing_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary["frame_timing_log"] = str(timing_path)
    summary["end_to_end_sec_including_initialization"] = (
        float(summary["initialization_sec"])
        + float(summary["streaming_wall_time_sec"])
    )

    if args.with_pose_metrics:
        from run_async_pipeline_metrics import evaluate_pose

        summary["pose_metrics"] = evaluate_pose(
            runner,
            resolved,
            int(summary["num_frames"]),
            work_dir,
        )

    output = work_dir / "staged_pipeline_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[staged pipeline summary]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
