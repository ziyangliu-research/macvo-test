from __future__ import annotations

import torch

from .backend_core import StreamingIncrementalBackend as _BaseStreamingBackend
from .contracts import LocalGaussianPacket
from .geometry import rotate_harmonics_local_to_world


class StreamingIncrementalBackend(_BaseStreamingBackend):
    """GraphDECO backend with appearance-preserving local-to-world SH alignment.

    The v1 backend already transforms means and Gaussian orientation correctly,
    but it kept higher-order SH coefficients in the canonical left-camera basis.
    GraphDECO evaluates SH against world-space view directions, so those
    coefficients must be rotated by the same c2w rotation used for the packet.
    """

    def _packet_to_graphdeco(
        self,
        packet: LocalGaussianPacket,
        T_world_from_left: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | bool]]:
        tensors, reset_stats = super()._packet_to_graphdeco(
            packet, T_world_from_left
        )

        non_blocking = packet.harmonics.device.type == "cpu" and packet.harmonics.is_pinned()
        harmonics_local = packet.harmonics.to(
            self.device, dtype=torch.float32, non_blocking=non_blocking
        )
        T = T_world_from_left.to(self.device, dtype=torch.float32)
        harmonics_world = rotate_harmonics_local_to_world(harmonics_local, T)

        target_coeffs = (self.config.sh_degree + 1) ** 2
        source_coeffs = int(harmonics_world.shape[-1])
        if source_coeffs < target_coeffs:
            pad = torch.zeros(
                harmonics_world.shape[0],
                3,
                target_coeffs - source_coeffs,
                device=self.device,
                dtype=harmonics_world.dtype,
            )
            harmonics_world = torch.cat([harmonics_world, pad], dim=-1)
        elif source_coeffs > target_coeffs:
            harmonics_world = harmonics_world[..., :target_coeffs]

        tensors["f_dc"] = harmonics_world[:, :, 0].unsqueeze(1).contiguous()
        tensors["f_rest"] = (
            harmonics_world[:, :, 1:].permute(0, 2, 1).contiguous()
        )
        return tensors, reset_stats
