#!/usr/bin/env python3
"""Run the reproducible pipeline with pre-maintenance opacity census diagnostics.

This wrapper does not change optimization, opacity reset, densification, pruning,
or historical replay.  It records why Gaussians survive the current GraphDECO
opacity-pruning rule before each maintenance event, then delegates to
``run_pipeline_execution_benchmark_repro.py``.

The key distinction is between the current ReSplat packet and the previously
accumulated map.  Before the first maintenance of a packet, no structural map
change has happened since append, so the final ``current_packet_count`` rows are
still exactly the newly inserted packet.  We therefore log opacity and
visibility statistics separately for:

- all Gaussians;
- historical / pre-existing Gaussians;
- current-packet Gaussians.

``denom > 0`` is used as the GraphDECO densification-stat visibility signal for
the first optimization half.  In the canonical 100-iteration / maintenance@50
setup this tells us whether a Gaussian participated in at least one visible
update during the first 50 recent-view iterations.

Results are written to ``opacity_census_log.json`` in the backend output
directory.  The wrapper is diagnostic only; it calls the original maintenance
unchanged after recording the census.
"""
from __future__ import annotations

import math
from typing import Any

import torch


_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def _install_opacity_census() -> None:
    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_initialize_gaussians = StreamingIncrementalBackend._initialize_gaussians
    original_append_gaussians = StreamingIncrementalBackend._append_gaussians
    original_maintenance = StreamingIncrementalBackend._run_maintenance

    if getattr(original_maintenance, "_opacity_census_patch", False):
        return

    def initialize_with_packet_count(
        self: StreamingIncrementalBackend,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        setattr(self, "_opacity_census_current_packet_count", int(tensors["xyz"].shape[0]))
        original_initialize_gaussians(self, tensors)

    def append_with_packet_count(
        self: StreamingIncrementalBackend,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        setattr(self, "_opacity_census_current_packet_count", int(tensors["xyz"].shape[0]))
        original_append_gaussians(self, tensors)

    def _tensor_stats(
        opacity: torch.Tensor,
        denom: torch.Tensor,
        reset_ceiling: float,
    ) -> dict[str, Any]:
        opacity = opacity.reshape(-1)
        denom = denom.reshape(-1)
        count = int(opacity.numel())
        if count == 0:
            return {"count": 0}

        visible = denom > 0
        never_visible = ~visible
        reset_eps = max(1e-6, abs(reset_ceiling) * 1e-4)
        at_or_below_reset = opacity <= (reset_ceiling + reset_eps)
        grew_above_reset = opacity > (reset_ceiling + reset_eps)
        prune_threshold = 0.005
        between_prune_and_reset = torch.logical_and(
            opacity >= prune_threshold,
            opacity <= (reset_ceiling + reset_eps),
        )

        quantile_values = torch.quantile(
            opacity.float(),
            torch.tensor(_QUANTILES, device=opacity.device, dtype=torch.float32),
        )
        quantiles = {
            f"q{int(round(q * 100)):02d}": float(v.item())
            for q, v in zip(_QUANTILES, quantile_values)
        }

        thresholds: dict[str, Any] = {}
        for threshold in _THRESHOLDS:
            mask = opacity < threshold
            key = f"lt_{threshold:g}"
            n = int(mask.sum().item())
            thresholds[key] = {
                "count": n,
                "ratio": float(n / count),
            }

        visible_count = int(visible.sum().item())
        never_visible_count = count - visible_count
        never_visible_survive_005 = int(
            torch.logical_and(never_visible, opacity >= 0.005).sum().item()
        )
        never_visible_at_or_above_reset = int(
            torch.logical_and(never_visible, opacity >= (reset_ceiling - reset_eps)).sum().item()
        )
        between_count = int(between_prune_and_reset.sum().item())
        grew_count = int(grew_above_reset.sum().item())
        at_or_below_count = int(at_or_below_reset.sum().item())

        return {
            "count": count,
            "opacity_mean": float(opacity.mean().item()),
            "opacity_quantiles": quantiles,
            "thresholds": thresholds,
            "visible_any_count": visible_count,
            "visible_any_ratio": float(visible_count / count),
            "never_visible_count": never_visible_count,
            "never_visible_ratio": float(never_visible_count / count),
            "never_visible_but_survives_lt_0.005_prune_count": never_visible_survive_005,
            "never_visible_but_survives_lt_0.005_prune_ratio": float(
                never_visible_survive_005 / count
            ),
            "never_visible_at_or_above_reset_ceiling_count": never_visible_at_or_above_reset,
            "never_visible_at_or_above_reset_ceiling_ratio": float(
                never_visible_at_or_above_reset / count
            ),
            "opacity_between_0.005_and_reset_ceiling_count": between_count,
            "opacity_between_0.005_and_reset_ceiling_ratio": float(between_count / count),
            "opacity_at_or_below_reset_ceiling_count": at_or_below_count,
            "opacity_at_or_below_reset_ceiling_ratio": float(at_or_below_count / count),
            "opacity_grew_above_reset_ceiling_count": grew_count,
            "opacity_grew_above_reset_ceiling_ratio": float(grew_count / count),
            "denom_mean": float(denom.float().mean().item()),
            "denom_max": float(denom.max().item()),
        }

    def maintenance_with_opacity_census(
        self: StreamingIncrementalBackend,
        update,
        local_iteration: int,
        radii: torch.Tensor,
    ):
        g = self.gaussians
        total_count = int(g.get_xyz.shape[0])
        current_count = int(getattr(self, "_opacity_census_current_packet_count", 0))
        current_count = min(max(current_count, 0), total_count)
        historical_count = total_count - current_count

        opacity = g.get_opacity.detach().reshape(-1)
        denom = g.denom.detach().reshape(-1)
        if int(denom.numel()) != total_count:
            raise RuntimeError(
                f"opacity census denom size {denom.numel()} != Gaussian count {total_count}"
            )

        reset_ceiling = float(self.config.new_packet_reset_max_opacity)
        all_stats = _tensor_stats(opacity, denom, reset_ceiling)
        historical_stats = _tensor_stats(
            opacity[:historical_count],
            denom[:historical_count],
            reset_ceiling,
        )
        current_stats = _tensor_stats(
            opacity[historical_count:],
            denom[historical_count:],
            reset_ceiling,
        )

        entry: dict[str, Any] = {
            "frame_index": int(update.descriptor.frame_index),
            "train_packet_count": int(self.train_packet_count),
            "global_iteration": int(self.global_iteration),
            "local_iteration": int(local_iteration),
            "num_gaussians_pre_maintenance": total_count,
            "historical_gaussian_count": historical_count,
            "current_packet_gaussian_count": current_count,
            "reset_new_packet_opacity": bool(self.config.reset_new_packet_opacity),
            "new_packet_reset_max_opacity": reset_ceiling,
            "maintenance_min_opacity": float(self.config.maintenance_min_opacity),
            "all_gaussians": all_stats,
            "historical_gaussians": historical_stats,
            "current_packet_gaussians": current_stats,
        }

        log = getattr(self, "_opacity_census_log", None)
        if log is None:
            log = []
            setattr(self, "_opacity_census_log", log)
        log.append(entry)
        if self.config.write_runtime_artifacts:
            self._save_json("opacity_census_log.json", log)

        current = current_stats
        print(
            "[opacity census] "
            f"frame={entry['frame_index']} packet={entry['train_packet_count']} "
            f"G={total_count} current={current_count} "
            f"current_lt005={current.get('thresholds', {}).get('lt_0.005', {}).get('ratio')} "
            f"current_005_to_reset={current.get('opacity_between_0.005_and_reset_ceiling_ratio')} "
            f"current_grew_above_reset={current.get('opacity_grew_above_reset_ceiling_ratio')} "
            f"current_never_visible={current.get('never_visible_ratio')} "
            f"current_never_visible_survives={current.get('never_visible_but_survives_lt_0.005_prune_ratio')}",
            flush=True,
        )

        event = original_maintenance(self, update, local_iteration, radii)
        entry["num_gaussians_post_maintenance"] = int(g.get_xyz.shape[0])
        entry["maintenance_count_delta"] = (
            entry["num_gaussians_post_maintenance"] - total_count
        )
        if self.config.write_runtime_artifacts:
            self._save_json("opacity_census_log.json", log)
        return event

    initialize_with_packet_count._opacity_census_patch = True  # type: ignore[attr-defined]
    append_with_packet_count._opacity_census_patch = True  # type: ignore[attr-defined]
    maintenance_with_opacity_census._opacity_census_patch = True  # type: ignore[attr-defined]

    StreamingIncrementalBackend._initialize_gaussians = initialize_with_packet_count
    StreamingIncrementalBackend._append_gaussians = append_with_packet_count
    StreamingIncrementalBackend._run_maintenance = maintenance_with_opacity_census

    print(
        "[opacity census] installed; training and maintenance policy unchanged",
        flush=True,
    )


def main() -> None:
    _install_opacity_census()

    # Import after patch installation so the reproducibility wrapper can layer
    # historical replay and S0-S4 diagnostics around these diagnostic hooks.
    import run_pipeline_execution_benchmark_repro as repro

    repro.main()


if __name__ == "__main__":
    main()
