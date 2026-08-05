import unittest
from pathlib import Path

import torch

from async_pipeline.contracts import (
    FrameDescriptor,
    LocalGaussianPacket,
    Observation,
    PoseEstimate,
)
from async_pipeline.joiner import OrderedFrontendJoiner


def descriptor(i: int, is_test: bool = False) -> FrameDescriptor:
    return FrameDescriptor(i, i, i, Path(f"L{i}.png"), Path(f"R{i}.png"), is_test)


def observation(d: FrameDescriptor) -> Observation:
    return Observation(d, torch.zeros(3, 4, 4), torch.eye(3))


def pose(d: FrameDescriptor) -> PoseEstimate:
    return PoseEstimate(d, torch.eye(4))


def packet(d: FrameDescriptor) -> LocalGaussianPacket:
    return LocalGaussianPacket(
        descriptor=d,
        means=torch.zeros(1, 3),
        scales=torch.ones(1, 3),
        rotations_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        harmonics=torch.zeros(1, 3, 1),
        opacities=torch.tensor([0.5]),
        context_intrinsics=torch.eye(3).repeat(2, 1, 1),
        context_extrinsics=torch.eye(4).repeat(2, 1, 1),
    )


class JoinerTest(unittest.TestCase):
    def test_out_of_order_frontends_emit_in_sequence_order(self) -> None:
        d0, d1 = descriptor(0), descriptor(1)
        joiner = OrderedFrontendJoiner()
        joiner.register(d0)
        joiner.register(d1)
        self.assertEqual(joiner.add_observation(observation(d0)), [])
        self.assertEqual(joiner.add_observation(observation(d1)), [])
        self.assertEqual(joiner.add_pose(pose(d1)), [])
        self.assertEqual(joiner.add_packet(packet(d1)), [])
        self.assertEqual(joiner.add_packet(packet(d0)), [])
        first = joiner.add_pose(pose(d0))
        self.assertEqual([u.descriptor.sequence_index for u in first], [0, 1])

    def test_test_frame_never_accepts_packet(self) -> None:
        d = descriptor(0, is_test=True)
        joiner = OrderedFrontendJoiner([d])
        with self.assertRaises(ValueError):
            joiner.add_packet(packet(d))

    def test_test_frame_emits_without_packet(self) -> None:
        d = descriptor(0, is_test=True)
        joiner = OrderedFrontendJoiner([d])
        self.assertEqual(joiner.add_observation(observation(d)), [])
        updates = joiner.add_pose(pose(d))
        self.assertEqual(len(updates), 1)
        self.assertIsNone(updates[0].packet)


if __name__ == "__main__":
    unittest.main()
