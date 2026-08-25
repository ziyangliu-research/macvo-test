#!/usr/bin/env python3
"""Re-evaluate locally visible Gaussians by resetting their opacity each packet.

Experimental branch wrapper.  The baseline pipeline is intentionally preserved:

- a dense ReSplat packet is appended at every train timestamp;
- the active camera window remains ``backend.local_map_size`` (normally 10);
- each packet keeps the same total optimization budget (normally 100 iters);
- densify/prune maintenance remains at the configured local iteration (normally 50);
- fixed-budget historical replay remains *after* maintenance, exactly as in the
  existing reproducibility wrapper (normally 80 recent + 20 historical iters).

The only added operation is performed after the new packet and camera have been
inserted, immediately before local optimization starts:

1. Render the current global Gaussian map from every camera in the active local
   window.
2. Take the union of renderer-defined visible Gaussians (``visibility_filter``).
3. Cap/reset the opacity of every Gaussian in that union to the configured reset
   value (by default ``backend.new_packet_reset_max_opacity``, normally 0.01).
4. Zero the row-wise Adam opacity moments for those Gaussians.  This makes the
   reset comparable to newly appended Gaussian rows, whose optimizer moments are
   initialized to zero, instead of allowing old Adam momentum to immediately
   undo the reset.

No Gaussian is protected from the normal maintenance threshold.  The existing
empty-map-safe replay guard is retained, so aggressive pruning can still reduce
an intermediate map to zero without crashing diff-gaussian-rasterization.

Important semantic note: ``visibility_filter`` is the GraphDECO rasterizer's
visibility definition (projected/rasterized Gaussian with positive visibility),
not a contribution/occlusion-aware importance score.  A later experiment can
replace this with a stricter contribution criterion if needed.

Optional environment variable:

``PIPELINE_LOCAL_VISIBLE_RESET_MAX_OPACITY``
    Override the reset cap.  If omitted, use
    ``backend.new_packet_reset_max_opacity``.
"""
from __future__ import annotations

import os
import time
from typing import Any, Sequence

import torch

import run_pipeline_execution_benchmark_repro as repro
import run_pipeline_execution_benchmark_repro_empty_safe as empty_safe


def _reset_local_visible_opacity(
    self: Any,
    update: Any,
    active_cameras: Sequence[Any],
) -> dict[str, Any]:
    """Reset opacity for the union of Gaussians visible in ``active_cameras``."""

    total = int(self.gaussians.get_xyz.shape[0])
    reset_cap_raw = os.environ.get("PIPELINE_LOCAL_VISIBLE_RESET_MAX_OPACITY")
    reset_cap = (
        float(reset_cap_raw)
        if reset_cap_raw is not None
        else float(self.config.new_packet_reset_max_opacity)
    )
    if not 0.0 < reset_cap < 1.0:
        raise ValueError(
            "local-visible opacity reset cap must be strictly between 0 and 1, "
            f"got {reset_cap}"
        )

    start = time.perf_counter()
    visible_mask = torch.zeros(total, dtype=torch.bool, device=self.device)
    per_view_visible: list[int] = []

    # Visibility is measured on the just-appended global map, before any local
    # optimization for this packet.  Use the deterministic configured background;
    # background color does not affect the rasterizer visibility mask.
    with torch.no_grad():
        for camera in active_cameras:
            render_pkg = self.render(
                camera,
                self.gaussians,
                self.pipe,
                self.background,
                use_trained_exp=False,
                separate_sh=False,
            )
            indices = self._visibility_indices(
                render_pkg["visibility_filter"],
                total,
            )
            per_view_visible.append(int(indices.numel()))
            if indices.numel() > 0:
                visible_mask[indices] = True

        visible_count = int(visible_mask.sum().item())

        if visible_count > 0:
            opacity_before_all = torch.sigmoid(self.gaussians._opacity.detach())
            selected_before = opacity_before_all[visible_mask]

            # Mirror the baseline new-packet reset semantics: cap opacity rather
            # than increasing already-lower opacity values.
            selected_after = torch.minimum(
                selected_before,
                torch.full_like(selected_before, reset_cap),
            )
            eps = 1e-6
            self.gaussians._opacity[visible_mask] = torch.logit(
                selected_after.clamp(eps, 1.0 - eps)
            )

            # New packet rows enter Adam with zero moments.  Selectively clear the
            # same row-wise state for mature Gaussians that are re-evaluated here.
            # Keep scalar/global state such as Adam's step counter unchanged.
            opacity_param = self.gaussians._opacity
            optimizer_state = self.gaussians.optimizer.state.get(opacity_param)
            state_tensors_zeroed: list[str] = []
            if optimizer_state is not None:
                for key, value in optimizer_state.items():
                    if (
                        torch.is_tensor(value)
                        and value.ndim >= 1
                        and int(value.shape[0]) == total
                    ):
                        value[visible_mask] = 0
                        state_tensors_zeroed.append(str(key))

            before_mean = float(selected_before.mean().item())
            before_min = float(selected_before.min().item())
            before_max = float(selected_before.max().item())
            before_above_005 = float((selected_before >= 0.005).float().mean().item())
            before_above_001 = float((selected_before >= 0.01).float().mean().item())
            before_above_0050 = float((selected_before >= 0.05).float().mean().item())
            after_mean = float(selected_after.mean().item())
            after_max = float(selected_after.max().item())
        else:
            state_tensors_zeroed = []
            before_mean = before_min = before_max = 0.0
            before_above_005 = before_above_001 = before_above_0050 = 0.0
            after_mean = after_max = 0.0

    visibility_and_reset_sec = self._sync_elapsed(start)

    event: dict[str, Any] = {
        "frame_index": int(update.descriptor.frame_index),
        "train_packet_count": int(self.train_packet_count),
        "global_iteration_before_optimization": int(self.global_iteration),
        "local_window_num_views": len(active_cameras),
        "total_gaussians_before_reset": total,
        "visible_union_count": visible_count,
        "visible_union_fraction": (0.0 if total == 0 else visible_count / total),
        "per_view_visible_counts": per_view_visible,
        "reset_max_opacity": reset_cap,
        "visible_opacity_before_mean": before_mean,
        "visible_opacity_before_min": before_min,
        "visible_opacity_before_max": before_max,
        "visible_fraction_opacity_ge_0.005": before_above_005,
        "visible_fraction_opacity_ge_0.01": before_above_001,
        "visible_fraction_opacity_ge_0.05": before_above_0050,
        "visible_opacity_after_mean": after_mean,
        "visible_opacity_after_max": after_max,
        "opacity_optimizer_state_tensors_zeroed": state_tensors_zeroed,
        "visibility_and_reset_sec": visibility_and_reset_sec,
    }

    log = getattr(self, "_local_visible_opacity_reset_log", None)
    if log is None:
        log = []
        setattr(self, "_local_visible_opacity_reset_log", log)
    log.append(event)

    if self.config.write_runtime_artifacts:
        self._save_json("local_visible_opacity_reset_log.json", log)

    if self.wandb_run is not None:
        # The next optimizer iteration increments global_iteration by one.  Log at
        # that step so reset diagnostics align with the packet's first train step.
        self.wandb_run.log(
            {
                "local_reset/visible_union_count": visible_count,
                "local_reset/visible_union_fraction": event["visible_union_fraction"],
                "local_reset/reset_max_opacity": reset_cap,
                "local_reset/opacity_before_mean": before_mean,
                "local_reset/opacity_after_mean": after_mean,
                "local_reset/visibility_and_reset_sec": visibility_and_reset_sec,
                "local_reset/local_window_num_views": len(active_cameras),
                "stream/frame_index": int(update.descriptor.frame_index),
                "stream/train_packet_count": int(self.train_packet_count),
            },
            step=int(self.global_iteration) + 1,
        )

    print(
        "[local-visible opacity reset] "
        f"packet={self.train_packet_count} "
        f"frame={update.descriptor.frame_index} "
        f"views={len(active_cameras)} "
        f"visible={visible_count}/{total} "
        f"({event['visible_union_fraction']:.3f}) "
        f"opacity_mean={before_mean:.5f}->{after_mean:.5f} "
        f"cap={reset_cap:.5f} "
        f"sec={visibility_and_reset_sec:.4f}",
        flush=True,
    )
    return event


def _install_replay_then_local_visible_reset() -> None:
    """Keep the empty-safe replay implementation, then prepend local reset."""

    # First install the exact replay implementation already used by the current
    # aggressive-pruning experiments.  This preserves maintenance/replay order.
    empty_safe._install_historical_replay_empty_safe()

    from async_pipeline.backend_core import StreamingIncrementalBackend

    replay_optimize = StreamingIncrementalBackend._optimize_active_map
    if getattr(replay_optimize, "_local_visible_opacity_reset_patch", False):
        return

    def optimize_with_local_visible_reset(
        self: StreamingIncrementalBackend,
        update,
        active_cameras,
    ):
        _reset_local_visible_opacity(self, update, active_cameras)
        return replay_optimize(self, update, active_cameras)

    optimize_with_local_visible_reset._local_visible_opacity_reset_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = optimize_with_local_visible_reset

    print(
        "[experiment] local-visible opacity reset enabled before optimization; "
        "maintenance and historical replay timing are unchanged",
        flush=True,
    )


# repro.main() installs replay only when PIPELINE_HISTORICAL_REPLAY_FRACTION > 0.
# Replace that installer with the composed replay + local-visible-reset variant.
repro._install_historical_replay = _install_replay_then_local_visible_reset


if __name__ == "__main__":
    replay_raw = os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION")
    if replay_raw is None or float(replay_raw) <= 0.0:
        raise RuntimeError(
            "This experiment wrapper intentionally preserves the historical replay "
            "baseline. Set PIPELINE_HISTORICAL_REPLAY_FRACTION, e.g. 0.2."
        )
    repro.main()
