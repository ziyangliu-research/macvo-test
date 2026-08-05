from __future__ import annotations

from typing import Any

import torch

from .contracts import FrameDescriptor, LocalGaussianPacket


def packet_from_resplat_gaussians(
    descriptor: FrameDescriptor,
    gaussians: Any,
    context: dict[str, torch.Tensor],
    *,
    inference_sec: float,
    metadata: dict[str, Any] | None = None,
    move_to_cpu: bool = True,
    pin_memory: bool = False,
) -> LocalGaussianPacket:
    """Build the minimal queue packet directly from a ReSplat Gaussians object."""

    def first_batch(value: torch.Tensor | None, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"ReSplat output is missing required field {name}")
        if value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(f"{name} must include batch dimension 1, got {tuple(value.shape)}")
        return value[0].detach().contiguous()

    packet = LocalGaussianPacket(
        descriptor=descriptor,
        means=first_batch(gaussians.means, "means"),
        scales=first_batch(gaussians.scales, "scales"),
        # ReSplat build_covariance explicitly interprets this field as xyzw.
        rotations_xyzw=first_batch(gaussians.rotations, "rotations"),
        harmonics=first_batch(gaussians.harmonics, "harmonics"),
        opacities=first_batch(gaussians.opacities, "opacities").reshape(-1),
        context_intrinsics=context["intrinsics"][0].detach().contiguous(),
        context_extrinsics=context["extrinsics"][0].detach().contiguous(),
        inference_sec=float(inference_sec),
        metadata={} if metadata is None else dict(metadata),
    )
    packet.validate()
    return packet.cpu(pin_memory=pin_memory) if move_to_cpu else packet
