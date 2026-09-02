#!/usr/bin/env python3
"""Post-hoc global refinement that CONTINUES the online Gaussian optimizer.

Compared with the earlier pilot runner, this version does not call
``training_setup`` at the online/global boundary.  Adam state for the final online
map is therefore retained.  If opacity reset is enabled, the backend's native
``reset_opacity`` is used, which updates the opacity parameter through the
existing optimizer while leaving the other Gaussian optimizer groups untouched.

Refinement budget is expressed as full passes over all mapping/train cameras.
Each pass shuffles all train cameras and visits every camera exactly once.
No densification or pruning is performed during this post-hoc stage.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _disable_intermediate_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation


def _install_refinement() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl
    passes = int(os.environ.get("PIPELINE_GLOBAL_REFINE_PASSES", "30"))
    eval_every = int(os.environ.get("PIPELINE_GLOBAL_REFINE_EVAL_EVERY", "500"))
    reset_opacity = _env_flag("PIPELINE_GLOBAL_REFINE_RESET_OPACITY", True)
    seed = int(os.environ.get("PIPELINE_GLOBAL_REFINE_SEED", os.environ.get("PIPELINE_BENCHMARK_SEED", "0")))

    if passes <= 0 or eval_every <= 0:
        raise ValueError("global refinement passes/eval_every must be positive")

    def write_curve(self, curve: list[dict[str, Any]]) -> None:
        jp = self.output_dir / "posthoc_global_refinement_curve.json"
        cp = self.output_dir / "posthoc_global_refinement_curve.csv"
        jp.write_text(json.dumps(curve, indent=2), encoding="utf-8")
        fields = [
            "stage", "refinement_iteration", "equivalent_passes", "num_gaussians",
            "train_psnr", "train_ssim", "train_l1", "test_psnr", "test_ssim", "test_l1",
            "cumulative_optimization_sec", "block_optimization_sec",
            "optimization_view_updates_per_sec", "metric_evaluation_sec",
        ]
        with cp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in curve:
                w.writerow({k: row.get(k) for k in fields})

    def make_row(self, stage, it, train, test, opt_sec, block_sec, eval_sec, ntrain):
        return {
            "stage": stage,
            "refinement_iteration": int(it),
            "equivalent_passes": 0.0 if ntrain == 0 else float(it) / float(ntrain),
            "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
            "train_psnr": train.get("psnr"), "train_ssim": train.get("ssim"), "train_l1": train.get("l1"),
            "test_psnr": test.get("psnr"), "test_ssim": test.get("ssim"), "test_l1": test.get("l1"),
            "cumulative_optimization_sec": float(opt_sec),
            "block_optimization_sec": block_sec,
            "optimization_view_updates_per_sec": None if opt_sec <= 0 or it <= 0 else float(it) / float(opt_sec),
            "metric_evaluation_sec": eval_sec,
        }

    def print_row(row):
        print(
            "[posthoc-global-refine/continue-opt] "
            f"iter={row['refinement_iteration']:5d} passes={row['equivalent_passes']:.2f} "
            f"Train={row.get('train_psnr')} / {row.get('train_ssim')} "
            f"Test={row.get('test_psnr')} / {row.get('test_ssim')} "
            f"G={row['num_gaussians']} upd/s={row.get('optimization_view_updates_per_sec')}",
            flush=True,
        )

    def finalize_with_refinement(self):
        if self.train_packet_count <= 0 or self.gaussians is None or int(self.gaussians.get_xyz.shape[0]) == 0:
            return original_finalize_impl(self)

        from utils.loss_utils import l1_loss, ssim

        ntrain = len(self.train_cameras)
        ntest = len(self.test_cameras)
        total = passes * ntrain
        online_iteration_end = int(self.global_iteration)
        g_before = int(self.gaussians.get_xyz.shape[0])

        print(
            "\n[posthoc-global-refine/continue-opt] "
            f"train_views={ntrain} passes={passes} total={total} eval_every={eval_every} "
            f"reset_opacity={reset_opacity} G={g_before}", flush=True,
        )

        t0 = time.perf_counter()
        online_train = self._evaluate(self.train_cameras)
        online_test = self._evaluate(self.test_cameras)
        baseline_eval_sec = self._sync_elapsed(t0)
        curve = [make_row(self, "online_before_global_refine", 0, online_train, online_test, 0.0, None, baseline_eval_sec, ntrain)]
        print_row(curve[-1])
        write_curve(self, curve)

        reset_stats = {"enabled": False}
        if reset_opacity:
            before = self.gaussians.get_opacity.detach().clone()
            if not hasattr(self.gaussians, "reset_opacity"):
                raise RuntimeError("GaussianModel has no reset_opacity(); cannot perform optimizer-safe opacity reset")
            # Native GraphDECO reset is optimizer-aware.  It resets/replaces only
            # the opacity parameter group; Adam state for xyz/SH/scale/rotation is retained.
            self.gaussians.reset_opacity()
            after = self.gaussians.get_opacity.detach()
            reset_stats = {
                "enabled": True,
                "implementation": "GaussianModel.reset_opacity()",
                "before_mean": float(before.mean().item()),
                "before_max": float(before.max().item()),
                "after_mean": float(after.mean().item()),
                "after_max": float(after.max().item()),
            }
            print("[posthoc-global-refine/continue-opt] opacity reset:", reset_stats, flush=True)

        # IMPORTANT: no training_setup() here.  Continue the online Adam optimizer.
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        rng = random.Random(seed)
        refine_it = 0
        cumulative_opt_sec = 0.0
        block_start = time.perf_counter()

        for _pass in range(passes):
            order = list(range(ntrain))
            rng.shuffle(order)
            for camera_index in order:
                refine_it += 1
                self.global_iteration += 1
                # Match MonoGS CR semantics: keep Adam state but restart the
                # refinement-stage xyz LR index from 1.
                self.gaussians.update_learning_rate(refine_it)

                camera = self.train_cameras[camera_index]
                background = torch.rand(3, device=self.device) if self.opt.random_background else self.background
                pkg = self.render(camera, self.gaussians, self.pipe, background, use_trained_exp=False, separate_sh=False)
                image = pkg["render"]
                gt = camera.original_image
                if camera.alpha_mask is not None:
                    image = image * camera.alpha_mask
                ll1 = l1_loss(image, gt)
                ssim_value = ssim(image, gt)
                loss = (1.0 - self.opt.lambda_dssim) * ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
                loss.backward()
                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                checkpoint = refine_it % eval_every == 0
                is_final = refine_it == total
                if not checkpoint and not is_final:
                    continue

                block_sec = self._sync_elapsed(block_start)
                cumulative_opt_sec += block_sec

                if not is_final:
                    te = time.perf_counter()
                    tr = self._evaluate(self.train_cameras)
                    ts = self._evaluate(self.test_cameras)
                    eval_sec = self._sync_elapsed(te)
                    row = make_row(self, "posthoc_global_refine", refine_it, tr, ts, cumulative_opt_sec, block_sec, eval_sec, ntrain)
                    curve.append(row)
                    print_row(row)
                    write_curve(self, curve)
                    block_start = time.perf_counter()

        # If total is itself a checkpoint, block timing above already captured it.
        # Otherwise the final iteration also captured the tail block because is_final=True.
        summary = original_finalize_impl(self)
        fm = summary.get("final_metrics") or {}
        final_train = fm.get("train_inserted") or {}
        final_test = fm.get("test_all") or {}
        final_row = make_row(self, "posthoc_global_refine_final", total, final_train, final_test, cumulative_opt_sec, None, None, ntrain)
        if curve and int(curve[-1].get("refinement_iteration", -1)) == total:
            curve[-1] = final_row
        else:
            curve.append(final_row)
        print_row(final_row)
        write_curve(self, curve)

        refine_summary = {
            "enabled": True,
            "protocol": "post-hoc full-map global refinement, continued online optimizer",
            "online_iteration_end": online_iteration_end,
            "num_train_views": ntrain,
            "num_test_views": ntest,
            "passes": passes,
            "total_refinement_iterations": total,
            "eval_every_iterations": eval_every,
            "each_train_view_updates": passes,
            "camera_sampling": "random permutation of every training camera per pass",
            "continue_online_optimizer": True,
            "restart_refinement_xyz_lr_index": True,
            "opacity_reset": reset_stats,
            "topology_fixed": True,
            "densification": False,
            "pruning": False,
            "gaussians_before_refinement": g_before,
            "gaussians_after_refinement": int(self.gaussians.get_xyz.shape[0]),
            "optimization_only_sec": cumulative_opt_sec,
            "optimization_view_updates_per_sec": None if cumulative_opt_sec <= 0 else total / cumulative_opt_sec,
            "online_before_refinement": {"train_inserted": online_train, "test_all": online_test},
            "final_after_refinement": {"train_inserted": final_train, "test_all": final_test},
        }
        summary["posthoc_global_refinement"] = refine_summary
        self._save_json("incremental_backend_summary.json", summary)
        self._save_json("posthoc_global_refinement_summary.json", refine_summary)
        return summary

    BackendEvaluationMixin._finalize_impl = finalize_with_refinement


if __name__ == "__main__":
    _disable_intermediate_evaluation()
    _install_refinement()
    safe.repro.main()
