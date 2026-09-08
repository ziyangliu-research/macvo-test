#!/usr/bin/env python3
"""Timing-only runner for multi-checkpoint post-hoc global refinement.

No image-quality metrics are rendered. One opacity reset is followed by one
continuous refinement trajectory to max(checkpoints), default 30 passes. CUDA is
synchronized at each checkpoint so cumulative refinement-only wall times for
10/15/20/25/30 passes are directly available without metric-render overhead.
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


def _parse_checkpoints() -> tuple[int, ...]:
    raw = os.environ.get("PIPELINE_GLOBAL_REFINE_CHECKPOINT_PASSES", "10,15,20,25,30")
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"global-refine checkpoint must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("PIPELINE_GLOBAL_REFINE_CHECKPOINT_PASSES is empty")
    return tuple(sorted(set(values)))


def _install_timing_refinement() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl
    checkpoints = _parse_checkpoints()
    max_pass = max(checkpoints)
    seed = int(
        os.environ.get(
            "PIPELINE_GLOBAL_REFINE_SEED",
            os.environ.get("PIPELINE_BENCHMARK_SEED", "0"),
        )
    )

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
        if ntrain <= 0:
            return original_finalize_impl(self)

        online_iteration_end = int(self.global_iteration)
        g_before = int(self.gaussians.get_xyz.shape[0])

        _sync(self)
        refinement_start = time.perf_counter()

        if not hasattr(self.gaussians, "reset_opacity"):
            raise RuntimeError("GaussianModel has no reset_opacity()")
        self.gaussians.reset_opacity()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        rng = random.Random(seed)
        refinement_iteration = 0
        cumulative: dict[str, float] = {}

        for pass_index in range(1, max_pass + 1):
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

            if pass_index in checkpoints:
                _sync(self)
                elapsed = time.perf_counter() - refinement_start
                cumulative[str(pass_index)] = elapsed
                print(
                    "[timing-only/global-refine] "
                    f"checkpoint_pass={pass_index} iterations={refinement_iteration} "
                    f"cumulative_sec={elapsed:.3f}",
                    flush=True,
                )

        _sync(self)
        total_refinement_sec = time.perf_counter() - refinement_start

        summary = original_finalize_impl(self)
        segments: dict[str, float] = {}
        previous_pass = 0
        previous_time = 0.0
        for checkpoint in checkpoints:
            current_time = cumulative[str(checkpoint)]
            segments[f"{previous_pass}_to_{checkpoint}"] = current_time - previous_time
            previous_pass = checkpoint
            previous_time = current_time

        timing = {
            "protocol": "post-hoc global refinement timing only; one continuous multi-checkpoint trajectory",
            "checkpoint_passes": list(checkpoints),
            "max_pass": max_pass,
            "num_train_views": ntrain,
            "each_train_view_updates": max_pass,
            "total_refinement_iterations": refinement_iteration,
            "online_iteration_end": online_iteration_end,
            "continue_online_optimizer": True,
            "opacity_reset": True,
            "topology_fixed": True,
            "densification": False,
            "pruning": False,
            "gaussians_before_refinement": g_before,
            "gaussians_after_refinement": int(self.gaussians.get_xyz.shape[0]),
            "checkpoint_cumulative_wall_time_sec": cumulative,
            "checkpoint_segment_wall_time_sec": segments,
            "refinement_wall_time_sec": total_refinement_sec,
        }
        summary["posthoc_global_refinement_timing"] = timing
        self._save_json("posthoc_global_refinement_timing_checkpoints.json", timing)
        self._save_json("incremental_backend_summary.json", summary)
        return summary

    BackendEvaluationMixin._finalize_impl = finalize_with_timing_refinement


if __name__ == "__main__":
    _disable_all_evaluation()
    _install_timing_refinement()
    safe.repro.main()
