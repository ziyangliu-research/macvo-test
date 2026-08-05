from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Optional

from .contracts import (
    BackendUpdate,
    FrameDescriptor,
    LocalGaussianPacket,
    Observation,
    PoseEstimate,
)


@dataclass
class _PendingFrame:
    descriptor: FrameDescriptor
    first_seen_monotonic: float = field(default_factory=time.monotonic)
    observation: Optional[Observation] = None
    pose: Optional[PoseEstimate] = None
    packet: Optional[LocalGaussianPacket] = None


class OrderedFrontendJoiner:
    """Join asynchronous pose and ReSplat results without reordering updates.

    Descriptors may be registered incrementally so the frame loader does not need
    to materialize the entire sequence before the workers start.
    """

    def __init__(self, descriptors: Iterable[FrameDescriptor] = ()) -> None:
        self._order: list[int] = []
        self._pending: Dict[int, _PendingFrame] = {}
        self._next_position = 0
        self._registration_closed = False
        for descriptor in descriptors:
            self.register(descriptor)
        if self._order:
            self.close_registration()

    def register(self, descriptor: FrameDescriptor) -> None:
        if self._registration_closed:
            raise RuntimeError("descriptor registration is already closed")
        expected = 0 if not self._order else self._order[-1] + 1
        if descriptor.sequence_index != expected:
            raise ValueError(
                "sequence_index values must be contiguous from zero; "
                f"expected {expected}, got {descriptor.sequence_index}"
            )
        if descriptor.sequence_index in self._pending:
            raise ValueError(f"duplicate descriptor {descriptor.sequence_index}")
        self._order.append(descriptor.sequence_index)
        self._pending[descriptor.sequence_index] = _PendingFrame(descriptor=descriptor)

    def close_registration(self) -> None:
        self._registration_closed = True

    @property
    def finished(self) -> bool:
        return (
            self._registration_closed
            and self._next_position >= len(self._order)
        )

    def add_observation(self, value: Observation) -> list[BackendUpdate]:
        value.validate()
        slot = self._slot(value.descriptor)
        if slot.observation is not None:
            raise ValueError(f"duplicate observation for sequence {value.descriptor.sequence_index}")
        slot.observation = value
        return list(self._drain_ready())

    def add_pose(self, value: PoseEstimate) -> list[BackendUpdate]:
        value.validate()
        if not value.committed:
            return []
        slot = self._slot(value.descriptor)
        if slot.pose is not None and value.revision <= slot.pose.revision:
            raise ValueError(
                f"non-increasing pose revision for sequence {value.descriptor.sequence_index}: "
                f"existing={slot.pose.revision}, new={value.revision}"
            )
        slot.pose = value
        return list(self._drain_ready())

    def add_packet(self, value: LocalGaussianPacket) -> list[BackendUpdate]:
        value.validate()
        slot = self._slot(value.descriptor)
        if slot.descriptor.is_test:
            raise ValueError(
                "strict split violation: ReSplat packet was generated for test frame "
                f"{value.descriptor.frame_index}"
            )
        if slot.packet is not None:
            raise ValueError(f"duplicate packet for sequence {value.descriptor.sequence_index}")
        slot.packet = value
        return list(self._drain_ready())

    def _slot(self, descriptor: FrameDescriptor) -> _PendingFrame:
        try:
            slot = self._pending[descriptor.sequence_index]
        except KeyError as exc:
            raise KeyError(
                f"sequence_index {descriptor.sequence_index} was not registered"
            ) from exc
        if slot.descriptor != descriptor:
            raise ValueError("descriptor identity mismatch for the same sequence_index")
        return slot

    def _drain_ready(self) -> Iterator[BackendUpdate]:
        while self._next_position < len(self._order):
            sequence_index = self._order[self._next_position]
            slot = self._pending[sequence_index]
            ready = slot.observation is not None and slot.pose is not None
            if slot.descriptor.is_test:
                ready = ready and slot.packet is None
            else:
                ready = ready and slot.packet is not None
            if not ready:
                return
            assert slot.observation is not None
            assert slot.pose is not None
            update = BackendUpdate(
                descriptor=slot.descriptor,
                observation=slot.observation,
                pose=slot.pose,
                packet=slot.packet,
                join_wait_sec=max(0.0, time.monotonic() - slot.first_seen_monotonic),
            )
            update.validate()
            self._next_position += 1
            yield update
