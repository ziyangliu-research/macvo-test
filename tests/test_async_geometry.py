import math
import unittest
from pathlib import Path

import torch

from async_pipeline.contracts import FrameDescriptor, LocalGaussianPacket
from async_pipeline.geometry import (
    align_local_packet_to_world,
    quaternion_xyzw_to_matrix,
    quaternion_xyzw_to_wxyz,
    rotate_local_quaternions_to_world_xyzw,
    transform_covariances,
)


def covariance(scales: torch.Tensor, rotations_xyzw: torch.Tensor) -> torch.Tensor:
    R = quaternion_xyzw_to_matrix(rotations_xyzw)
    S = torch.diag_embed(scales)
    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)


class GeometryContractTest(unittest.TestCase):
    def test_local_covariance_alignment_matches_world_quaternion(self) -> None:
        angle = math.radians(37.0)
        T = torch.eye(4)
        T[:3, :3] = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T[:3, 3] = torch.tensor([1.0, -2.0, 0.5])
        local_q = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.2, -0.1, 0.3, 0.9]],
            dtype=torch.float32,
        )
        local_q = local_q / torch.linalg.vector_norm(local_q, dim=-1, keepdim=True)
        scales = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.15, 0.07]])
        local_cov = covariance(scales, local_q)
        expected = transform_covariances(T, local_cov)
        world_q = rotate_local_quaternions_to_world_xyzw(local_q, T)
        actual = covariance(scales, world_q)
        self.assertTrue(torch.allclose(expected, actual, atol=2e-6, rtol=2e-6))

    def test_graphdeco_reorder(self) -> None:
        identity_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(
            torch.equal(
                quaternion_xyzw_to_wxyz(identity_xyzw),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )

    def test_packet_alignment_keeps_scale_appearance(self) -> None:
        descriptor = FrameDescriptor(
            sequence_index=0,
            frame_index=0,
            timestamp_ns=0,
            left_path=Path("left.png"),
            right_path=Path("right.png"),
            is_test=False,
        )
        packet = LocalGaussianPacket(
            descriptor=descriptor,
            means=torch.tensor([[1.0, 2.0, 3.0]]),
            scales=torch.tensor([[0.1, 0.2, 0.3]]),
            rotations_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            harmonics=torch.zeros((1, 3, 9)),
            opacities=torch.tensor([0.7]),
            context_intrinsics=torch.eye(3).repeat(2, 1, 1),
            context_extrinsics=torch.eye(4).repeat(2, 1, 1),
        )
        T = torch.eye(4)
        T[:3, 3] = torch.tensor([4.0, 5.0, 6.0])
        aligned = align_local_packet_to_world(packet, T)
        self.assertTrue(torch.equal(aligned.means, torch.tensor([[5.0, 7.0, 9.0]])))
        self.assertTrue(torch.equal(aligned.scales, packet.scales))
        self.assertTrue(torch.equal(aligned.harmonics, packet.harmonics))
        self.assertTrue(torch.equal(aligned.opacities, packet.opacities))


if __name__ == "__main__":
    unittest.main()
