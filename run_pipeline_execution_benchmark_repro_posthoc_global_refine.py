#!/usr/bin/env python3
"""Run the reproducible online pipeline followed by post-hoc global refinement.

Online mapping is unchanged.  After the final online update, this wrapper:

1. evaluates the online-only map once;
2. applies a one-time global max-opacity reset (default 0.01);
3. rebuilds a fresh GraphDECO optimizer / LR schedule on the existing map;
4. runs a fixed number of shuffled full-training-view passes;
5. evaluates all Train/Test views every N refinement steps;
6. records optimization-only throughput separately from metric rendering.

The refinement stage deliberately keeps map topology fixed: no densification and
no pruning are performed.  This makes the iteration-count sweep interpretable
and keeps per-step cost approximately stationary for this pilot experiment.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _disable_intermediate_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original = BackendEvaluationMixin._record_evaluation
    if getattr(original, "_posthoc_global_refine_no_stream_eval", False):
        return

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    no_intermediate_evaluation._posthoc_global_refine_no_stream_eval = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation
    print(
        "[posthoc-global-refine] intermediate online metric rendering disabled",
        flush=True,
    )


def _install_posthoc_global_refinement() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl
    if getattr(original_finalize_impl, "_posthoc_global_refine_patch", False):
        return

    passes = int(os.environ.get("PIPELINE_GLOBAL_REFINE_PASSES", "5"))
    eval_every = int(os.environ.get("PIPELINE_GLOBAL_REFINE_EVAL_EVERY", "100"))
    reset_max_opacity = float(
        os.environ.get("PIPELINE_GLOBAL_REFINE_RESET_OPACITY_MAX", "0.01")
    )
    seed = int(
        os.environ.get(
            "PIPELINE_GLOBAL_REFINE_SEED",
            os.environ.get("PIPELINE_BENCHMARK_SEED", "0"),
        )
    )

    if passes <= 0:
        raise ValueError("PIPELINE_GLOBAL_REFINE_PASSES must be positive")
    if eval_every <= 0:
        raise ValueError("PIPELINE_GLOBAL_REFINE_EVAL_EVERY must be positive")
    if not 0.0 < reset_max_opacity < 1.0:
        raise ValueError(
            "PIPELINE_GLOBAL_REFINE_RESET_OPACITY_MAX must lie in (0,1)"
        )

    def _write_curve(self, curve: list[dict[str, Any]]) -> None:
        json_path = self.output_dir / "posthoc_global_refinement_curve.json"
        csv_path = self.output_dir / "posthoc_global_refinement_curve.csv"
        json_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")

        fields = [
            "stage",
            "refinement_iteration",
            "equivalent_passes",
            "num_gaussians",
            "train_psnr",
            "train_ssim",
            "train_l1",
            "test_psnr",
            "test_ssim",
            "test_l1",
            "cumulative_optimization_sec",
            "block_optimization_sec",
            "optimization_view_updates_per_sec",
            "metric_evaluation_sec",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in curve:
                writer.writerow({key: row.get(key) for key in fields})

    def _metric_row(
        self,
        *,
        stage: str,
        refinement_iteration: int,
        train_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        cumulative_optimization_sec: float,
        block_optimization_sec: float | None,
        metric_evaluation_sec: float | None,
        num_train_views: int,
    ) -> dict[str, Any]:
        speed = (
            None
            if cumulative_optimization_sec <= 0.0 or refinement_iteration <= 0
            else float(refinement_iteration) / cumulative_optimization_sec
        )
        return {
            "stage": stage,
            "refinement_iteration": int(refinement_iteration),
            "equivalent_passes": (
                0.0
                if num_train_views <= 0
                else float(refinement_iteration) / float(num_train_views)
            ),
            "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
            "train_psnr": train_metrics.get("psnr"),
            "train_ssim": train_metrics.get("ssim"),
            "train_l1": train_metrics.get("l1"),
            "test_psnr": test_metrics.get("psnr"),
            "test_ssim": test_metrics.get("ssim"),
            "test_l1": test_metrics.get("l1"),
            "cumulative_optimization_sec": float(cumulative_optimization_sec),
            "block_optimization_sec": block_optimization_sec,
            "optimization_view_updates_per_sec": speed,
            "metric_evaluation_sec": metric_evaluation_sec,
        }

    def _print_row(row: dict[str, Any]) -> None:
        print(
            "[posthoc-global-refine] "
            f"iter={row['refinement_iteration']:4d} "
            f"passes={row['equivalent_passes']:.3f} "
            f"Train={row.get('train_psnr')} / {row.get('train_ssim')} "
            f"Test={row.get('test_psnr')} / {row.get('test_ssim')} "
            f"G={row['num_gaussians']} "
            f"opt_view_updates_per_sec={row.get('optimization_view_updates_per_sec')}",
            flush=True,
        )

    def finalize_with_global_refinement(self):
        if (
            self.train_packet_count <= 0
            or self.gaussians is None
            or int(self.gaussians.get_xyz.shape[0]) <= 0
        ):
            return original_finalize_impl(self)

        from utils.loss_utils import l1_loss, ssim

        num_train_views = len(self.train_cameras)
        total_refine_iterations = passes * num_train_views
        online_iteration_end = int(self.global_iteration)
        gaussians_before = int(self.gaussians.get_xyz.shape[0])

        print(
            "\n[posthoc-global-refine] starting: "
            f"train_views={num_train_views}, passes={passes}, "
            f"total_iterations={total_refine_iterations}, eval_every={eval_every}, "
            f"G={gaussians_before}",
            flush=True,
        )

        # Paired online-only baseline from the exact same run, before reset.
        baseline_eval_start = time.perf_counter()
        online_train = self._evaluate(self.train_cameras)
        online_test = self._evaluate(self.test_cameras)
        baseline_eval_sec = self._sync_elapsed(baseline_eval_start)

        curve: list[dict[str, Any]] = []
        baseline_row = _metric_row(
            self,
            stage="online_before_global_reset",
            refinement_iteration=0,
            train_metrics=online_train,
            test_metrics=online_test,
            cumulative_optimization_sec=0.0,
            block_optimization_sec=None,
            metric_evaluation_sec=baseline_eval_sec,
            num_train_views=num_train_views,
        )
        curve.append(baseline_row)
        _print_row(baseline_row)
        _write_curve(self, curve)

        # Standard 3DGS-style max-opacity reset, but performed explicitly so the
        # semantics are stable even if the local GraphDECO checkout differs.
        with torch.no_grad():
            opacity_before = self.gaussians.get_opacity.detach().clone()
            capped = torch.minimum(
                opacity_before,
                torch.full_like(opacity_before, reset_max_opacity),
            )
            eps = 1e-6
            reset_internal = torch.logit(capped.clamp(eps, 1.0 - eps))
            self.gaussians._opacity = torch.nn.Parameter(
                reset_internal.detach().clone().requires_grad_(True)
            )

        opacity_after = self.gaussians.get_opacity.detach()
        opacity_reset_stats = {
            "maximum_opacity": reset_max_opacity,
            "before_mean": float(opacity_before.mean().item()),
            "before_max": float(opacity_before.max().item()),
            "after_mean": float(opacity_after.mean().item()),
            "after_max": float(opacity_after.max().item()),
        }
        print(
            "[posthoc-global-refine] global opacity reset: "
            f"mean {opacity_reset_stats['before_mean']:.6f} -> "
            f"{opacity_reset_stats['after_mean']:.6f}, "
            f"max {opacity_reset_stats['before_max']:.6f} -> "
            f"{opacity_reset_stats['after_max']:.6f}",
            flush=True,
        )

        # Treat the post-hoc stage as a separate optimizer run on the final map.
        # Parameters are preserved; Adam moments and the position LR schedule are
        # restarted from refinement iteration 1.
        self.gaussians.training_setup(self.opt)
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        rng = random.Random(seed)
        refinement_iteration = 0
        cumulative_optimization_sec = 0.0
        block_start = time.perf_counter()
        last_timed_iteration = 0

        for pass_index in range(passes):
            order = list(range(num_train_views))
            rng.shuffle(order)
            for camera_index in order:
                refinement_iteration += 1
                self.global_iteration += 1
                self.gaussians.update_learning_rate(refinement_iteration)

                camera = self.train_cameras[camera_index]
                background = (
                    torch.rand(3, device=self.device)
                    if self.opt.random_background
                    else self.background
                )
                render_pkg = self.render(
                    camera,
                    self.gaussians,
                    self.pipe,
                    background,
                    use_trained_exp=False,
                    separate_sh=False,
                )
                image = render_pkg["render"]
                gt = camera.original_image
                if camera.alpha_mask is not None:
                    image = image * camera.alpha_mask
                ll1 = l1_loss(image, gt)
                ssim_value = ssim(image, gt)
                loss = (
                    (1.0 - self.opt.lambda_dssim) * ll1
                    + self.opt.lambda_dssim * (1.0 - ssim_value)
                )
                loss.backward()

                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                checkpoint = refinement_iteration % eval_every == 0
                is_final = refinement_iteration == total_refine_iterations
                if not checkpoint and not is_final:
                    continue

                block_opt_sec = self._sync_elapsed(block_start)
                cumulative_optimization_sec += block_opt_sec
                last_timed_iteration = refinement_iteration

                # Avoid evaluating the final point twice.  The normal backend
                # finalize below will compute the canonical final Train/Test metrics.
                if not is_final:
                    eval_start = time.perf_counter()
                    train_metrics = self._evaluate(self.train_cameras)
                    test_metrics = self._evaluate(self.test_cameras)
                    eval_sec = self._sync_elapsed(eval_start)
                    row = _metric_row(
                        self,
                        stage="posthoc_global_refine",
                        refinement_iteration=refinement_iteration,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        cumulative_optimization_sec=cumulative_optimization_sec,
                        block_optimization_sec=block_opt_sec,
                        metric_evaluation_sec=eval_sec,
                        num_train_views=num_train_views,
                    )
                    curve.append(row)
                    _print_row(row)
                    _write_curve(self, curve)
                    block_start = time.perf_counter()
                else:
                    block_start = time.perf_counter()

        # This only matters when the last iteration was not itself a checkpoint.
        if last_timed_iteration < total_refine_iterations:
            tail_sec = self._sync_elapsed(block_start)
            cumulative_optimization_sec += tail_sec

        # Preserve the normal final artifact/summary behavior.  Its final metrics
        # are the canonical refinement endpoint and are reused as the last curve row.
        summary = original_finalize_impl(self)
        final_metrics = summary.get("final_metrics") or {}
        final_train = final_metrics.get("train_inserted") or {}
        final_test = final_metrics.get("test_all") or {}

        final_row = _metric_row(
            self,
            stage="posthoc_global_refine_final",
            refinement_iteration=total_refine_iterations,
            train_metrics=final_train,
            test_metrics=final_test,
            cumulative_optimization_sec=cumulative_optimization_sec,
            block_optimization_sec=None,
            metric_evaluation_sec=None,
            num_train_views=num_train_views,
        )
        if not curve or int(curve[-1]["refinement_iteration"]) != total_refine_iterations:
            curve.append(final_row)
        else:
            curve[-1] = final_row
        _print_row(final_row)
        _write_curve(self, curve)

        refine_summary = {
            "enabled": True,
            "protocol": "post-hoc global full-map parameter refinement",
            "online_iteration_end": online_iteration_end,
            "num_train_views": num_train_views,
            "num_test_views": len(self.test_cameras),
            "passes": passes,
            "total_refinement_iterations": total_refine_iterations,
            "eval_every_iterations": eval_every,
            "camera_sampling": "each pass is a random permutation of all training cameras",
            "each_train_view_updates": passes,
            "opacity_reset": opacity_reset_stats,
            "fresh_optimizer_and_lr_schedule": True,
            "topology_fixed": True,
            "densification": False,
            "pruning": False,
            "gaussians_before_refinement": gaussians_before,
            "gaussians_after_refinement": int(self.gaussians.get_xyz.shape[0]),
            "optimization_only_sec": cumulative_optimization_sec,
            "optimization_view_updates_per_sec": (
                None
                if cumulative_optimization_sec <= 0.0
                else total_refine_iterations / cumulative_optimization_sec
            ),
            "curve_json": str(self.output_dir / "posthoc_global_refinement_curve.json"),
            "curve_csv": str(self.output_dir / "posthoc_global_refinement_curve.csv"),
            "online_before_reset": {
                "train_inserted": online_train,
                "test_all": online_test,
            },
            "final_after_refinement": {
                "train_inserted": final_train,
                "test_all": final_test,
            },
        }
        summary["posthoc_global_refinement"] = refine_summary
        self._save_json("incremental_backend_summary.json", summary)
        self._save_json("posthoc_global_refinement_summary.json", refine_summary)

        print(
            "[posthoc-global-refine] done: "
            f"optimization_only_sec={cumulative_optimization_sec:.3f}, "
            f"view_updates_per_sec={refine_summary['optimization_view_updates_per_sec']}",
            flush=True,
        )
        return summary

    finalize_with_global_refinement._posthoc_global_refine_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_with_global_refinement


if __name__ == "__main__":
    _disable_intermediate_evaluation()
    _install_posthoc_global_refinement()
    safe.repro.main()
