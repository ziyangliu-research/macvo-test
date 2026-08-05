from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Protocol

from .contracts import (
    BackendUpdate,
    FrameDescriptor,
    LocalGaussianPacket,
    Observation,
    PoseEstimate,
    StereoFrameInput,
    StopSignal,
    WorkerFailure,
)
from .joiner import OrderedFrontendJoiner
from .macvo_runtime import MacvoPoseFrontend
from .resplat_runtime import ResplatPacketGenerator
from .scheduler import QueuePolicy, ThreadWorker


class IncrementalBackend(Protocol):
    def initialize(self) -> None: ...

    def process(self, update: BackendUpdate) -> None: ...

    def finalize(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AsyncPipelineConfig:
    split_every: int = 5
    split_offset: int = 4
    split_index_mode: str = "local_index"
    queue_size: int = 2
    poll_timeout_sec: float = 0.05
    initialization_timeout_sec: float = 600.0
    summary_path: Path = Path("outputs/async_pipeline_summary.json")

    def __post_init__(self) -> None:
        if self.split_every <= 0:
            raise ValueError("split_every must be positive")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if self.poll_timeout_sec <= 0:
            raise ValueError("poll_timeout_sec must be positive")
        if self.initialization_timeout_sec <= 0:
            raise ValueError("initialization_timeout_sec must be positive")
        if self.split_index_mode not in {
            "local_index",
            "packet_index",
            "frame_index",
        }:
            raise ValueError(f"unknown split_index_mode {self.split_index_mode}")

    def is_test(self, descriptor: FrameDescriptor) -> bool:
        if self.split_index_mode in {"local_index", "packet_index"}:
            value = descriptor.sequence_index
        else:
            value = descriptor.frame_index
        return (value - self.split_offset) % self.split_every == 0


class NullValidationBackend:
    """Backend used to validate ordering and strict split without mapping."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def initialize(self) -> None:
        return

    def process(self, update: BackendUpdate) -> None:
        update.validate()
        self.updates.append(
            {
                "sequence_index": update.descriptor.sequence_index,
                "frame_index": update.descriptor.frame_index,
                "is_test": update.descriptor.is_test,
                "has_packet": update.packet is not None,
                "num_gaussians": 0 if update.packet is None else update.packet.num_gaussians,
                "join_wait_sec": update.join_wait_sec,
            }
        )

    def finalize(self) -> dict[str, Any]:
        return {
            "backend": "NullValidationBackend",
            "num_updates": len(self.updates),
            "updates": self.updates,
        }


class AsyncPipelineRunner:
    """Class-based bounded pipeline with persistent in-process components."""

    def __init__(
        self,
        config: AsyncPipelineConfig,
        pose_frontend: MacvoPoseFrontend,
        packet_generator: ResplatPacketGenerator,
        backend: IncrementalBackend,
    ) -> None:
        self.config = config
        self.pose_frontend = pose_frontend
        self.packet_generator = packet_generator
        self.backend = backend
        policy = QueuePolicy(maxsize=config.queue_size)
        self.pose_worker: ThreadWorker[tuple[FrameDescriptor, Any], PoseEstimate] = (
            ThreadWorker(
                "pose-frontend",
                lambda task: self.pose_frontend.process(*task),
                policy=policy,
                setup=self.pose_frontend.initialize,
                teardown=self.pose_frontend.terminate,
            )
        )
        self.packet_worker: ThreadWorker[StereoFrameInput, LocalGaussianPacket] = (
            ThreadWorker(
                "resplat-frontend",
                lambda frame_input: self.packet_generator.infer(
                    frame_input,
                    output_frame="left_camera_local",
                ).packet,
                policy=policy,
                setup=self.packet_generator.initialize,
                teardown=self.packet_generator.close,
            )
        )
        self.backend_worker: ThreadWorker[BackendUpdate, Any] = ThreadWorker(
            "incremental-backend",
            self.backend.process,
            policy=policy,
            setup=self.backend.initialize,
            teardown=self.backend.finalize,
        )
        self._pose_stopped = False
        self._packet_stopped = False
        self._backend_stopped = False
        self._backend_summary: Optional[dict[str, Any]] = None

    def run(self) -> dict[str, Any]:
        init_start = time.perf_counter()
        joiner = OrderedFrontendJoiner()
        # Initialize sequentially to avoid Hydra/import/CUDA allocator races,
        # while still constructing each model in the thread that owns it.
        for worker in (
            self.pose_worker,
            self.packet_worker,
            self.backend_worker,
        ):
            worker.start()
            worker.wait_ready(self.config.initialization_timeout_sec)
        initialization_sec = time.perf_counter() - init_start
        wall_start = time.perf_counter()

        submitted_train = 0
        submitted_test = 0
        emitted_updates = 0
        num_frames = 0
        try:
            for descriptor, frame, stereo_input, observation in self.pose_frontend.iter_frames():
                descriptor = replace(
                    descriptor,
                    is_test=self.config.is_test(descriptor),
                )
                stereo_input.descriptor = descriptor
                observation.descriptor = descriptor
                joiner.register(descriptor)
                num_frames += 1
                emitted_updates += self._submit_updates(
                    joiner.add_observation(observation)
                )
                # Both tasks are made runnable before the coordinator waits for
                # either result. ReSplat consumes the already decoded stereo pair.
                self.pose_worker.submit((descriptor, frame))
                if descriptor.is_test:
                    submitted_test += 1
                else:
                    self.packet_worker.submit(stereo_input)
                    submitted_train += 1
                emitted_updates += self._drain_frontends(joiner, block=False)
                self._drain_backend(block=False)

            joiner.close_registration()
            self.pose_worker.close_input()
            self.packet_worker.close_input()
            while not (self._pose_stopped and self._packet_stopped and joiner.finished):
                emitted_updates += self._drain_frontends(joiner, block=True)
                self._drain_backend(block=False)

            self.backend_worker.close_input()
            self.pose_worker.join()
            self.packet_worker.join()
            while not self._backend_stopped:
                self._drain_backend(block=True)
            self.backend_worker.join()
        except BaseException:
            self.pose_worker.request_stop()
            self.packet_worker.request_stop()
            self.backend_worker.request_stop()
            raise

        summary = {
            "pipeline": "asynchronous class-based MAC-VO + local ReSplat + incremental backend",
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
                "model_initialization": "once per component in its owning worker thread",
                "frontend_io": "one decoded stereo frame shared by MAC-VO/ReSplat adapters",
                "packet_handoff": self.packet_generator.config.handoff_mode,
                "backend_order": "strict sequence order after pose/packet join",
            },
            "queue_size": self.config.queue_size,
            "initialization_sec": initialization_sec,
            "streaming_wall_time_sec": time.perf_counter() - wall_start,
            "backend": self._backend_summary,
        }
        output = self.config.summary_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _submit_updates(self, updates: list[BackendUpdate]) -> int:
        for update in updates:
            self.backend_worker.submit(update)
        return len(updates)

    def _drain_frontends(self, joiner: OrderedFrontendJoiner, *, block: bool) -> int:
        emitted = 0
        timeout = self.config.poll_timeout_sec if block else 0.0
        queues = [self.pose_worker.output, self.packet_worker.output]
        received = False
        for output_queue in queues:
            try:
                item = output_queue.get(timeout=timeout if not received else 0.0)
            except queue.Empty:
                continue
            received = True
            if isinstance(item, WorkerFailure):
                raise RuntimeError(
                    f"worker {item.worker} failed: {item.message}\n{item.traceback_text}"
                )
            if isinstance(item, StopSignal):
                if item.source == "pose-frontend":
                    self._pose_stopped = True
                elif item.source == "resplat-frontend":
                    self._packet_stopped = True
                continue
            if isinstance(item, PoseEstimate):
                emitted += self._submit_updates(joiner.add_pose(item))
            elif isinstance(item, LocalGaussianPacket):
                emitted += self._submit_updates(joiner.add_packet(item))
            else:
                raise TypeError(f"unexpected frontend output type {type(item)}")
        return emitted

    def _drain_backend(self, *, block: bool) -> None:
        timeout = self.config.poll_timeout_sec if block else 0.0
        try:
            item = self.backend_worker.output.get(timeout=timeout)
        except queue.Empty:
            return
        if isinstance(item, WorkerFailure):
            raise RuntimeError(
                f"worker {item.worker} failed: {item.message}\n{item.traceback_text}"
            )
        if isinstance(item, StopSignal):
            self._backend_stopped = True
            return
        if isinstance(item, dict):
            self._backend_summary = item
            return
        if item is not None:
            raise TypeError(f"unexpected backend output type {type(item)}")
