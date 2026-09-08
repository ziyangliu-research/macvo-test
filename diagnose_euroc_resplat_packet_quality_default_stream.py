#!/usr/bin/env python3
"""Run the EuRoC standalone ReSplat packet diagnostic on CUDA's default stream.

ReSplat/custom CUDA operators in this project are known to produce severely
incorrect packet renders when the encoder is executed on a dedicated persistent
CUDA stream.  The formal serial benchmark already pins ReSplat to the default
stream.  This wrapper applies the same safeguard to the standalone EuRoC
packet-quality diagnostic before importing/running it.
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

from diagnose_euroc_resplat_packet_quality import main


if __name__ == "__main__":
    print("[diagnostic] ReSplat pinned to CUDA default stream", flush=True)
    main()
