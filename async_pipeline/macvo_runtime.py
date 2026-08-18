from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

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
    pose_source: str = "macvo"
    gt_pose_file: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise ValueError(
                f"invalid frame range [{self.start_index},{self.end_index})"
            )
        if self.pose_commit_policy not in {"one_frame_delayed", "end_of_sequence"}:
            raise ValueError(
                "pose_commit_policy must be one_frame_delayed or end_of_sequence"
            )
        if self.pose_source not in {"macvo", "gt"}:
            raise ValueError("pose_source must be macvo or gt")
        if self.pose_source == "gt" and self.gt_pose_file is None:
            raise ValueError("gt pose source requires gt_pose_file")


class MacvoPoseFrontend:
    """Frame loader plus either MAC-VO or exact TartanAir GT pose frontend.

    ``pose_source='macvo'`` preserves the normal online MAC-VO path.
    ``pose_source='gt'`` is an ablation mode: the same sequence loader and image
    preprocessing are retained, but MAC-VO is not constructed or run. Instead,
    poses are read from the configured TartanAir tx ty tz qx qy qz qw file and
    converted with the exact same TartanAir->OpenCV and first-frame-relative
    convention used by ``run_async_pipeline_metrics.py``.
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
        self._initialized = False
        self._gt_rows: Optional[np.ndarray] = None

    def initialize(self) -> None:
        if self._initialized:
            return
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

        from DataLoader import SequenceBase, StereoFrame, smart_transform
        from Utility.Config import load_config

        odom_cfg, _ = load_config(self.config.odom_config.expanduser().resolve())
        data_cfg, _ = load_config(self.config.data_config.expanduser().resolve())

        if not hasattr(odom_cfg, "Preprocess"):
            raise ValueError(
                f"MAC-VO odometry config has no Preprocess section: {self.config.odom_config}"
            )

        sequence = (
            SequenceBase[StereoFrame]
            .instantiate(data_cfg.type, data_cfg.args)
            .clip(self.config.start_index, self.config.end_index)
        )
        sequence = smart_transform(sequence, odom_cfg.Preprocess)
        if self.config.preload:
            sequence = sequence.preload()
        self.sequence = sequence

        if self.config.pose_source == "gt":
            assert self.config.gt_pose_file is not None
            gt_path = self.config.gt_pose_file.expanduser().resolve()
            if not gt_path.is_file():
                raise FileNotFoundError(f"GT pose file not found: {gt_path}")
            rows = np.loadtxt(gt_path, dtype=np.float64)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
            if rows.ndim != 2 or rows.shape[1] < 7:
                raise ValueError(
                    f"GT pose file must contain tx ty tz qx qy qz qw rows, got {rows.shape}"
                )
            if self.config.end_index > rows.shape[0]:
                raise IndexError(
                    f"requested frames [{self.config.start_index},{self.config.end_index}) "
                    f"outside GT pose file with {rows.shape[0]} rows"
                )
            self._gt_rows = rows
            first_absolute = self._gt_absolute_pose(self.config.start_index)
            self._T0_inv = torch.linalg.inv(first_absolute)
            self._initialized = True
            return

        if not hasattr(odom_cfg, "Odometry"):
            raise ValueError(
                f"MAC-VO odometry config has no Odometry section: {self.config.odom_config}"
            )
        if not hasattr(odom_cfg.Odometry, "frontend"):
            raise ValueError(
                "MAC-VO Odometry section has no frontend entry; the complete root "
                "experiment config must be supplied"
            )

        from Odometry.MACVO import MACVO

        # MACVO.from_config expects the complete root namespace and internally
        # reads cfg.Odometry. The previous implementation wrapped the complete
        # dictionary inside another Odometry key, producing cfg.Odometry.Odometry.
        self.system = MACVO[StereoFrame].from_config(odom_cfg)

        if torch.cuda.is_available() and self.config.dedicated_cuda_stream:
            current = torch.cuda.current_stream()
            self.stream = torch.cuda.Stream()
            self.stream.wait_stream(current)
        self._initialized = True

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
                    left_path=(
                        root
                        / self.config.left_subdir
                        / self.config.left_pattern.format(index=frame_index)
                    ),
                    right_path=(
                        root
                        / self.config.right_subdir
                        / self.config.right_pattern.format(index=frame_index)
                    ),
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
            intrinsic = frame.stereo.frame_K.detach().cpu().float().contiguous()
            stereo_input = StereoFrameInput(
                descriptor=descriptor,
                left_image=left,
                right_image=right,
                intrinsic_pixel=intrinsic,
                baseline_m=float(frame.stereo.frame_baseline),
            )
            stereo_input.validate()
            observation = Observation(
                descriptor=descriptor,
                image=left,
                intrinsic_pixel=intrinsic,
            )
            observation.validate()
            yield descriptor, frame, stereo_input, observation

    def process(self, descriptor: FrameDescriptor, frame: Any) -> list[PoseEstimate]:
        self.initialize()
        if self._terminated:
            raise RuntimeError("cannot process frames after terminate")
        expected_sequence_index = len(self._descriptors)
        if descriptor.sequence_index != expected_sequence_index:
            raise ValueError(
                "pose frames must be submitted in sequence order; "
                f"expected {expected_sequence_index}, got {descriptor.sequence_index}"
            )
        self._descriptors.append(descriptor)

        if self.config.pose_source == "gt":
            estimate = PoseEstimate(
                descriptor=descriptor,
                T_world_from_left=self._gt_relative_pose(descriptor.frame_index),
                valid=True,
                committed=True,
                revision=0,
                latency_sec=0.0,
                metadata={
                    "source": "TartanAir-GT",
                    "coordinate_convention": "metric OpenCV c2w, first selected frame identity",
                    "commit_policy": "immediate_gt_ablation",
                    "one_frame_delay": False,
                    "need_interp": False,
                },
            )
            estimate.validate()
            return [estimate]

        assert self.system is not None
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
        if self.config.pose_source == "gt":
            self._terminated = True
            return []

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

    def _gt_absolute_pose(self, frame_index: int) -> torch.Tensor:
        if self._gt_rows is None:
            raise RuntimeError("GT poses are not initialized")
        if frame_index < 0 or frame_index >= self._gt_rows.shape[0]:
            raise IndexError(
                f"GT frame index {frame_index} outside [0,{self._gt_rows.shape[0]})"
            )
        pose = _pose7_xyzw_to_matrix(
            torch.from_numpy(self._gt_rows[frame_index, :7].copy())
        )
        return pose @ _tartan_from_cv(dtype=torch.float64)

    def _gt_relative_pose(self, frame_index: int) -> torch.Tensor:
        if self._T0_inv is None:
            first = self._gt_absolute_pose(self.config.start_index)
            self._T0_inv = torch.linalg.inv(first)
        return (self._T0_inv @ self._gt_absolute_pose(frame_index)).float()

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
        assert self.system is not None
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
                    "one_frame_delay": (
                        self.config.pose_commit_policy == "one_frame_delayed"
                    ),
                    "need_interp": need_interp,
                },
            )
            estimate.validate()
            output.append(estimate)
            self._next_emit += 1
        return output
