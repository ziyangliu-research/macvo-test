#!/usr/bin/env python3
"""Compatibility-fixed local packet prefilter benchmark.

This keeps the v1 experiment unchanged and only supplies GraphDECO's temporary
``tmp_radii`` auxiliary state before direct opacity-only pruning.  Upstream
``GaussianModel.prune_points`` assumes this state was created by
``densify_and_prune``; our packet prefilter intentionally skips densification,
so v1 could fail when even a very small number of points crossed the opacity
threshold.
"""
from __future__ import annotations

import torch

import run_pipeline_execution_benchmark_packet_prefilter as impl


_original_make_model = impl.LocalPacketPrefilter._make_model


def _make_model_with_prune_aux_state(self, packet):
    g = _original_make_model(self, packet)
    # prune_points() slices tmp_radii together with all trainable tensors.  The
    # prefilter has no densification/screen-size pruning, so zero radii are the
    # neutral bookkeeping value and do not affect the opacity mask.
    g.tmp_radii = torch.zeros(
        int(g.get_xyz.shape[0]),
        device=self.device,
        dtype=torch.float32,
    )
    return g


impl.LocalPacketPrefilter._make_model = _make_model_with_prune_aux_state


if __name__ == "__main__":
    resolved = impl._peek_resolved_config()
    impl._install_packet_prefilter(resolved)
    impl.repro.main()
