from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import torch


@dataclass(frozen=True)
class FrameDescriptor:
    """Immutable identity and source metadata for one stereo timestamp."""

    sequence_index: int
    frame_index: int
    timestamp_ns: int
    left_path: Path
    right_path: Path
    is_test: bool

    def __post_init__(self) -> None:
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")


@dataclass
class StereoFrameInput:
    """Single decoded stereo input shared by the two front-end workers.

    MAC-VO retains its native ``StereoFrame`` object, while ReSplat consumes this
    CPU representation. This prevents a second image decode in the normal async
    path and keeps the neural-network-specific preprocessing inside each adapter.
    """

    descriptor: FrameDescriptor
    left_image: torch.Tensor
    right_image: torch.Tensor
    intrinsic_pixel: torch.Tensor
    baseline_m: float

    def validate(self, *, deep: bool = False) -> None:
        for name in ("left_image", "right_image"):
            image = getattr(self, name)
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"{name} must have shape [3,H,W], got {tuple(image.shape)}"
                )
            if deep and not torch.isfinite(image).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if self.left_image.shape != self.right_image.shape:
            raise ValueError(
                "left/right image shapes differ: "
                f"{tuple(self.left_image.shape)} vs {tuple(self.right_image.shape)}"
            )
        if self.intrinsic_pixel.shape != (3, 3):
            raise ValueError(
                "intrinsic_pixel must have shape [3,3], "
                f"got {tuple(self.intrinsic_pixel.shape)}"
            )
        if deep and not torch.isfinite(self.intrinsic_pixel).all():
            raise ValueError("intrinsic_pixel contains NaN or Inf")
        if self.baseline_m <= 0:
            raise ValueError(f"baseline_m must be positive, got {self.baseline_m}")


@dataclass
class Observation:
    """Left-camera RGB observation consumed by the incremental backend."""

    descriptor: FrameDescriptor
    image: torch.Tensor
    intrinsic_pixel: torch.Tensor

    def validate(self, *, deep: bool = False) -> None:
        if self.image.ndim != 3 or self.image.shape[0] != 3:
            raise ValueError(f"image must have shape [3,H,W], got {tuple(self.image.shape)}")
        if self.intrinsic_pixel.shape != (3, 3):
            raise ValueError(
                "intrinsic_pixel must have shape [3,3], "
                f"got {tuple(self.intrinsic_pixel.shape)}"
            )
        if deep and not torch.isfinite(self.image).all():
            raise ValueError("observation image contains NaN or Inf")
        if deep and not torch.isfinite(self.intrinsic_pixel).all():
            raise ValueError("observation intrinsic contains NaN or Inf")


@dataclass
class PoseEstimate:
    """Committed metric OpenCV camera-to-world pose for one timestamp."""

    descriptor: FrameDescriptor
    T_world_from_left: torch.Tensor
    valid: bool = True
    committed: bool = True
    revision: int = 0
    latency_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.T_world_from_left.shape != (4, 4):
            raise ValueError(
                "T_world_from_left must have shape [4,4], "
                f"got {tuple(self.T_world_from_left.shape)}"
            )
        if not torch.isfinite(self.T_world_from_left).all():
            raise ValueError("pose contains NaN or Inf")
        bottom = self.T_world_from_left[3]
        expected = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=bottom.dtype,
            device=bottom.device,
        )
        if not torch.allclose(bottom, expected, atol=1e-5, rtol=0.0):
            raise ValueError(f"invalid homogeneous bottom row: {bottom.tolist()}")
        R = self.T_world_from_left[:3, :3].float()
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        if not torch.allclose(R.transpose(0, 1) @ R, eye, atol=2e-3, rtol=2e-3):
            raise ValueError("pose rotation is not sufficiently orthonormal")
        det = torch.det(R)
        if not torch.isclose(det, torch.ones_like(det), atol=2e-3, rtol=2e-3):
            raise ValueError(f"pose rotation determinant is {float(det):.6f}, expected 1")


@dataclass
class LocalGaussianPacket:
    """Minimal ReSplat packet in the current left-camera coordinate frame.

    Tensor conventions:
      means:            [N,3]
      scales:           [N,3], activated positive axis scales
      rotations_xyzw:   [N,4], normalized quaternion in ReSplat/scipy order
      harmonics:        [N,3,C], ReSplat SH layout
      opacities:        [N], activated values in [0,1]

    Covariance matrices are intentionally omitted because they are redundant with
    scale + rotation and substantially increase queue traffic. ReSplat exports
    rotations in *xyzw* order; GraphDECO uses *wxyz*. The backend performs the
    explicit convention conversion after world alignment.
    """

    descriptor: FrameDescriptor
    means: torch.Tensor
    scales: torch.Tensor
    rotations_xyzw: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    context_intrinsics: torch.Tensor
    context_extrinsics: torch.Tensor
    inference_sec: float = 0.0
    coordinate_frame: Literal["left_camera_local", "world"] = "left_camera_local"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def validate(self, *, deep: bool = False) -> None:
        n = self.num_gaussians
        expected = {
            "means": (n, 3),
            "scales": (n, 3),
            "rotations_xyzw": (n, 4),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if deep and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if self.harmonics.ndim != 3 or tuple(self.harmonics.shape[:2]) != (n, 3):
            raise ValueError(
                "harmonics must have shape [N,3,C], "
                f"got {tuple(self.harmonics.shape)}"
            )
        if deep and not torch.isfinite(self.harmonics).all():
            raise ValueError("harmonics contains NaN or Inf")
        if self.opacities.reshape(-1).shape != (n,):
            raise ValueError(
                f"opacities must contain N={n} values, got {tuple(self.opacities.shape)}"
            )
        if deep and not torch.isfinite(self.opacities).all():
            raise ValueError("opacities contains NaN or Inf")
        if self.context_intrinsics.shape != (2, 3, 3):
            raise ValueError(
                "context_intrinsics must have shape [2,3,3], "
                f"got {tuple(self.context_intrinsics.shape)}"
            )
        if self.context_extrinsics.shape != (2, 4, 4):
            raise ValueError(
                "context_extrinsics must have shape [2,4,4], "
                f"got {tuple(self.context_extrinsics.shape)}"
            )
        if deep:
            if torch.any(self.scales <= 0):
                raise ValueError("activated scales must be positive")
            opacity = self.opacities.reshape(-1)
            if torch.any((opacity < 0) | (opacity > 1)):
                raise ValueError("activated opacities must lie in [0,1]")
            qnorm = torch.linalg.vector_norm(self.rotations_xyzw.float(), dim=-1)
            if not torch.allclose(qnorm, torch.ones_like(qnorm), atol=2e-3, rtol=2e-3):
                raise ValueError("packet rotations are not normalized xyzw quaternions")

    def cpu(self, pin_memory: bool = False) -> "LocalGaussianPacket":
        def move(value: torch.Tensor) -> torch.Tensor:
            out = value.detach().to(device="cpu", non_blocking=False).contiguous()
            return out.pin_memory() if pin_memory and torch.cuda.is_available() else out

        return LocalGaussianPacket(
            descriptor=self.descriptor,
            means=move(self.means),
            scales=move(self.scales),
            rotations_xyzw=move(self.rotations_xyzw),
            harmonics=move(self.harmonics),
            opacities=move(self.opacities),
            context_intrinsics=move(self.context_intrinsics),
            context_extrinsics=move(self.context_extrinsics),
            inference_sec=self.inference_sec,
            coordinate_frame=self.coordinate_frame,
            metadata=dict(self.metadata),
        )


@dataclass
class BackendUpdate:
    """Ordered update emitted once pose and packet dependencies are satisfied."""

    descriptor: FrameDescriptor
    observation: Observation
    pose: PoseEstimate
    packet: Optional[LocalGaussianPacket]
    join_wait_sec: float = 0.0

    @property
    def is_train_update(self) -> bool:
        return self.packet is not None and not self.descriptor.is_test

    @property
    def is_test_observation(self) -> bool:
        return self.packet is None and self.descriptor.is_test

    def validate(self) -> None:
        self.observation.validate()
        self.pose.validate()
        if self.observation.descriptor != self.descriptor:
            raise ValueError("observation descriptor does not match update descriptor")
        if self.pose.descriptor != self.descriptor:
            raise ValueError("pose descriptor does not match update descriptor")
        if self.descriptor.is_test:
            if self.packet is not None:
                raise ValueError("strict split violation: test update must not contain a packet")
        else:
            if self.packet is None:
                raise ValueError("train update is missing its ReSplat packet")
            self.packet.validate()
            if self.packet.descriptor != self.descriptor:
                raise ValueError("packet descriptor does not match update descriptor")


@dataclass(frozen=True)
class StopSignal:
    source: str


@dataclass
class WorkerFailure:
    worker: str
    message: str
    traceback_text: str
    context: Mapping[str, Any] = field(default_factory=dict)
