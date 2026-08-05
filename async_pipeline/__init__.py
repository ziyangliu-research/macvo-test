"""Class-based asynchronous MAC-VO -> ReSplat -> incremental 3DGS pipeline.

This package intentionally contains no command-to-command subprocess chaining.
External repositories are loaded as Python libraries by dedicated runtime classes.
"""

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
from .geometry import align_local_packet_to_world, fixed_tartanair_stereo_rig_cv
from .joiner import OrderedFrontendJoiner

__all__ = [
    "BackendUpdate",
    "FrameDescriptor",
    "LocalGaussianPacket",
    "Observation",
    "OrderedFrontendJoiner",
    "PoseEstimate",
    "StereoFrameInput",
    "StopSignal",
    "WorkerFailure",
    "align_local_packet_to_world",
    "fixed_tartanair_stereo_rig_cv",
]
