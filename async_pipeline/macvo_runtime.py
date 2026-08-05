from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import torch

from .contracts import (
    FrameDescriptor,
    Observation,
    PoseEstimate,
    StereoFrameInput,
)


def _quat_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q).clamp_min(1e-12)
    x, y, z, w = q.unbind()
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)


def _pose7_xyzw_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    pose = pose.detach().cpu().double().reshape(7)
    output = torch.eye(4, dtype=torch.float64)
    output[:3, :3] = _quat_xyzw_to_matrix(pose[3:7])
    output[:3, 3] = pose[:3]
    return output


def _tartan_from_cv(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    output = torch.eye(4, dtype=dtype)
    output[:3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
    )
    return output


@dataclass(frozen=True)
class MacvoRuntimeConfig:
    repo: Path
    odom_config: Path
    data_config: Path
    start_index: int
    end_index: int
    left_subdir: str = "image_lcam_front"
    right_subdir: str = "image_rcam_front"
    left_pattern: str = "{index:06d}_lcam_front.png"
    right_pattern: str = "{index:06d}_rcam_front.png"
    preload: bool = False
    pose_commit_policy: str = "one_frame_delayed"
    dedicated_cuda_stream: bool = True

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise ValueError(
                f"invalid frame range [{self.start_index},{self.end_index})"
            )
        if self.pose_commit_policy not in {"one_frame_delayed", "end_of_sequence"}:
            raise ValueError(
                "pose_commit_policy must be one_frame_delayed or end_of_sequence"
            )


class MacvoPoseFrontend:
    """Persistent in-process MAC-VO runtime with committed streaming poses.

    MAC-VO's two-frame optimizer writes the previous optimization result back at
    the beginning of the next pair. Therefore the default policy emits frame t-1
    after processing frame t. This preserves a stable pose without waiting for
    the complete sequence and naturally introduces the known one-frame delay.
    """

    def __init__(self, config: MacvoRuntimeConfig) -> None:
        self.config = config
        self.repo = config.repo.expanduser().resolve()
        self.system: Any = None
        self.sequence: Any = None
        self._descriptors: list[FrameDescriptor] = []
        self._next_emit = 0
        self._T0_inv: Optional[torch.Tensor] = None
        self._last_run_sec = 0.0
        self._terminated = False
        self.stream: Optional[torch.cuda.Stream] = None

    def initialize(self) -> None:
        if self.system is not None:
            return
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from DataLoader import SequenceBase, StereoFrame, smart_transform
        from Odometry.MACVO import MACVO
        from Utility.Config import asNamespace, load_config

        odom_cfg, odom_dict = load_config(self.config.odom_config.expanduser().resolve())
        data_cfg, _ = load_config(self.config.data_config.expanduser().resolve())
        sequence = smart_transform(
            SequenceBase[StereoFrame]
            .instantiate(data_cfg.type, data_cfg.args)
            .clip(self.config.start_index, self.config.end_index),
            odom_cfg.Preprocess,
        )
        if self.config.preload:
            sequence = sequence.preload()
        self.sequence = sequence
        self.system = MACVO[StereoFrame].from_config(
            asNamespace({"Odometry": odom_dict})
        )
        if torch.cuda.is_available() and self.config.dedicated_cuda_stream:
            current = torch.cuda.current_stream()
            self.stream = torch.cuda.Stream()
            self.stream.wait_stream(current)

    def descriptors(self) -> list[FrameDescriptor]:
        self.initialize()
        from Utility.Config import load_config

        data_cfg, _ = load_config(self.config.data_config.expanduser().resolve())
        root = Path(data_cfg.args.root).expanduser().resolve()
        descriptors: list[FrameDescriptor] = []
        for sequence_index, frame_index in enumerate(
            range(self.config.start_index, self.config.end_index)
        ):
            descriptors.append(
                FrameDescriptor(
                    sequence_index=sequence_index,
                    frame_index=frame_index,
                    timestamp_ns=frame_index,
                    left_path=root
                    / self.config.left_subdir
                    / self.config.left_pattern.format(index=frame_index),
                    right_path=root
                    / self.config.right_subdir
                    / self.config.right_pattern.format(index=frame_index),
                    is_test=False,
                )
            )
        return descriptors

    def iter_frames(
        self,
    ) -> Iterator[tuple[FrameDescriptor, Any, StereoFrameInput, Observation]]:
        self.initialize()
        descriptors = self.descriptors()
        assert self.sequence is not None
        for descriptor, frame in zip(descriptors, self.sequence):
            timestamp = int(frame.stereo.frame_ns)
            descriptor = FrameDescriptor(
                sequence_index=descriptor.sequence_index,
                frame_index=int(frame.frame_idx),
                timestamp_ns=timestamp,
                left_path=descriptor.left_path,
                right_path=descriptor.right_path,
                is_test=descriptor.is_test,
            )
            left = frame.stereo.imageL[0].detach().cpu().float().contiguous()
            right = frame.stereo.imageR[0].detach().cpu().float().contiguous()
            K = frame.stereo.frame_K.detach().cpu().float().contiguous()
            stereo_input = StereoFrameInput(
                descriptor=descriptor,
                left_image=left,
                right_image=right,
                intrinsic_pixel=K,
                baseline_m=float(frame.stereo.frame_baseline),
            )
            stereo_input.validate()
            observation = Observation(
                descriptor=descriptor,
                image=left,
                intrinsic_pixel=K,
            )
            observation.validate()
            yield descriptor, frame, stereo_input, observation

    def process(self, descriptor: FrameDescriptor, frame: Any) -> list[PoseEstimate]:
        self.initialize()
        if self._terminated:
            raise RuntimeError("cannot process frames after terminate")
        assert self.system is not None
        expected_sequence_index = len(self._descriptors)
        if descriptor.sequence_index != expected_sequence_index:
            raise ValueError(
                "MAC-VO frames must be submitted in sequence order; "
                f"expected {expected_sequence_index}, got {descriptor.sequence_index}"
            )
        self._descriptors.append(descriptor)
        start = time.perf_counter()
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.system.run(frame)
            self.stream.synchronize()
        else:
            self.system.run(frame)
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
        self._last_run_sec = time.perf_counter() - start
        if self.config.pose_commit_policy == "end_of_sequence":
            return []
        committed_count = max(0, len(self._descriptors) - 1)
        return self._emit_until(committed_count)

    def terminate(self) -> list[PoseEstimate]:
        if self._terminated:
            return []
        self.initialize()
        assert self.system is not None
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.system.terminate()
            self.stream.synchronize()
        else:
            self.system.terminate()
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
        self._terminated = True
        return self._emit_until(len(self._descriptors))

    def _body_pose7(self, index: int) -> torch.Tensor:
        import pypose as pp

        graph = self.system.graph
        sensor = pp.SE3(graph.frames.data["pose"].tensor[index])
        T_BS = pp.SE3(graph.frames.data["T_BS"].tensor[index])
        return (T_BS @ sensor @ T_BS.Inv()).tensor().detach().cpu()

    def _opencv_relative_pose(self, index: int) -> torch.Tensor:
        absolute = _pose7_xyzw_to_matrix(self._body_pose7(index)) @ _tartan_from_cv()
        if self._T0_inv is None:
            first = _pose7_xyzw_to_matrix(self._body_pose7(0)) @ _tartan_from_cv()
            self._T0_inv = torch.linalg.inv(first)
        return (self._T0_inv @ absolute).float()

    def _emit_until(self, count: int) -> list[PoseEstimate]:
        output: list[PoseEstimate] = []
        graph = self.system.graph
        while self._next_emit < count:
            index = self._next_emit
            descriptor = self._descriptors[index]
            need_interp = bool(
                graph.frames.data["need_interp"][index].detach().cpu().item()
            )
            estimate = PoseEstimate(
                descriptor=descriptor,
                T_world_from_left=self._opencv_relative_pose(index),
                valid=not need_interp,
                committed=True,
                revision=0,
                latency_sec=self._last_run_sec,
                metadata={
                    "source": "MAC-VO",
                    "coordinate_convention": "metric OpenCV c2w, first frame identity",
                    "commit_policy": self.config.pose_commit_policy,
                    "one_frame_delay": self.config.pose_commit_policy
                    == "one_frame_delayed",
                    "need_interp": need_interp,
                },
            )
            estimate.validate()
            output.append(estimate)
            self._next_emit += 1
        return output
