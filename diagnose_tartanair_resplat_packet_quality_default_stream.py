#!/usr/bin/env python3
"""Run the TartanAir standalone ReSplat packet diagnostic on CUDA's default stream.

The formal serial benchmark already uses this stream safeguard.  Keep the
standalone diagnostic on the same validated execution path so packet-quality
numbers are not corrupted by the known persistent non-default CUDA-stream bug.
"""
from __future__ import annotations

import torch

from async_pipeline.resplat_runtime import ResplatPacketGenerator


_original_initialize = ResplatPacketGenerator.initialize


def _initialize_on_default_stream(self: ResplatPacketGenerator) -> None:
    _original_initialize(self)
    if self.device.type == "cuda":
        self.stream = torch.cuda.default_stream(self.device)


ResplatPacketGenerator.initialize = _initialize_on_default_stream

from diagnose_tartanair_resplat_packet_quality import main


if __name__ == "__main__":
    print("[diagnostic] ReSplat pinned to CUDA default stream", flush=True)
    main()
