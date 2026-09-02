#!/usr/bin/env python3
"""Timing-only runner for the final 10-pass global refinement protocol.

No image-quality metrics are rendered. Online mapping is unchanged. At the online
endpoint the runner applies one optimizer-safe opacity reset, continues the
existing online Gaussian optimizer for K full shuffled passes over all train
views, and records refinement-only wall time. Frame-level timing from the normal
execution benchmark remains available for online FPS / online wall time.
"""
from __future__ import annotations

import os
import random
import time

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _disable_all_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation


def _install_timing_refinement() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl
    passes = int(os.environ.get("PIPELINE_GLOBAL_REFINE_PASSES", "10"))
    seed = int(
        os.environ.get(
            "PIPELINE_GLOBAL_REFINE_SEED",
            os.environ.get("PIPELINE_BENCHMARK_SEED", "0"),
        )
    )
    if passes <= 0:
        raise ValueError("PIPELINE_GLOBAL_REFINE_PASSES must be positive")

    def _sync(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        else:
            torch.cuda.current_stream(self.device).synchronize()

    def finalize_with_timing_refinement(self):
        if (
            self.train_packet_count <= 0
            or self.gaussians is None
            or int(self.gaussians.get_xyz.shape[0]) == 0
        ):
            return original_finalize_impl(self)

        from utils.loss_utils import l1_loss, ssim

        ntrain = len(self.train_cameras)
        online_iteration_end = int(self.global_iteration)
        g_before = int(self.gaussians.get_xyz.shape[0])

        # Make the boundary explicit: all online CUDA work is complete before the
        # post-hoc timer starts. The timer includes the opacity reset itself.
        _sync(self)
        refinement_start = time.perf_counter()

        if not hasattr(self.gaussians, "reset_opacity"):
            raise RuntimeError("GaussianModel has no reset_opacity()")
        self.gaussians.reset_opacity()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        rng = random.Random(seed)
        refinement_iteration = 0
        for _pass in range(passes):
            order = list(range(ntrain))
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

        _sync(self)
        refinement_sec = time.perf_counter() - refinement_start

        # Stock finalize is safe here because the timing shell disables evaluation.
        summary = original_finalize_impl(self)
        timing = {
            "protocol": "post-hoc global refinement timing only",
            "passes": passes,
            "num_train_views": ntrain,
            "each_train_view_updates": passes,
            "total_refinement_iterations": refinement_iteration,
            "online_iteration_end": online_iteration_end,
            "continue_online_optimizer": True,
            "opacity_reset": True,
            "topology_fixed": True,
            "densification": False,
            "pruning": False,
            "gaussians_before_refinement": g_before,
            "gaussians_after_refinement": int(self.gaussians.get_xyz.shape[0]),
            "refinement_wall_time_sec": refinement_sec,
        }
        summary["posthoc_global_refinement_timing"] = timing
        self._save_json("posthoc_global_refinement_timing.json", timing)
        self._save_json("incremental_backend_summary.json", summary)

        print(
            "[timing-only/global-refine] "
            f"train_views={ntrain} passes={passes} iterations={refinement_iteration} "
            f"refinement_sec={refinement_sec:.3f}",
            flush=True,
        )
        return summary

    BackendEvaluationMixin._finalize_impl = finalize_with_timing_refinement


if __name__ == "__main__":
    _disable_all_evaluation()
    _install_timing_refinement()
    safe.repro.main()
