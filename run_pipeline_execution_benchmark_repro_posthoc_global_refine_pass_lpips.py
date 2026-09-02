#!/usr/bin/env python3
"""Pass-based post-hoc global refinement with PSNR/SSIM/LPIPS evaluation.

Online mapping is unchanged. After the final online update this runner:
- evaluates all Train/Test views with PSNR, SSIM and LPIPS(VGG);
- performs one optimizer-safe global opacity reset;
- continues the existing online Gaussian optimizer (Adam state retained);
- runs a fixed number of shuffled full passes over all mapping/train views;
- evaluates all Train/Test views after every completed pass;
- keeps Gaussian topology fixed during post-hoc refinement.

The refinement budget is expressed in passes, not raw iterations. Therefore each
mapping/train view receives exactly one optimizer update per pass.
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import Any

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _disable_intermediate_online_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation


def _install_lpips_evaluator() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    try:
        from lpips import LPIPS
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS evaluation requires the Python package 'lpips'. "
            "Install it in the macvo_resplat environment with: pip install lpips"
        ) from exc

    @torch.no_grad()
    def evaluate_with_lpips(self, cameras):
        from utils.image_utils import psnr
        from utils.loss_utils import l1_loss, ssim

        selected = list(cameras)
        if self.config.eval_max_views > 0:
            selected = selected[: self.config.eval_max_views]
        if not selected:
            return {"num_views": 0}

        model = getattr(self, "_posthoc_lpips_vgg", None)
        if model is None:
            model = LPIPS(net="vgg").to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            setattr(self, "_posthoc_lpips_vgg", model)

        l1_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
        for camera in selected:
            image = self.render(
                camera,
                self.gaussians,
                self.pipe,
                self.background,
                use_trained_exp=False,
                separate_sh=False,
            )["render"].clamp(0.0, 1.0)
            gt = camera.original_image.clamp(0.0, 1.0)

            l1_sum += float(l1_loss(image, gt).mean().item())
            psnr_sum += float(psnr(image, gt).mean().item())
            ssim_sum += float(ssim(image, gt).mean().item())
            # Match ReSplat's evaluation convention: LPIPS-VGG with normalize=True
            # for input images in [0,1].
            lp = model(gt.unsqueeze(0), image.unsqueeze(0), normalize=True)
            lpips_sum += float(lp.mean().item())

        count = len(selected)
        return {
            "num_views": count,
            "l1": l1_sum / count,
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / count,
            "lpips": lpips_sum / count,
        }

    BackendEvaluationMixin._evaluate = evaluate_with_lpips


def _install_pass_refinement() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl
    passes = int(os.environ.get("PIPELINE_GLOBAL_REFINE_PASSES", "20"))
    seed = int(
        os.environ.get(
            "PIPELINE_GLOBAL_REFINE_SEED",
            os.environ.get("PIPELINE_BENCHMARK_SEED", "0"),
        )
    )
    if passes <= 0:
        raise ValueError("PIPELINE_GLOBAL_REFINE_PASSES must be positive")

    def write_curve(self, curve: list[dict[str, Any]]) -> None:
        json_path = self.output_dir / "posthoc_global_refinement_pass_metrics.json"
        csv_path = self.output_dir / "posthoc_global_refinement_pass_metrics.csv"
        json_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")
        fields = [
            "pass",
            "refinement_iteration",
            "train_num_views",
            "test_num_views",
            "train_psnr",
            "train_ssim",
            "train_lpips",
            "test_psnr",
            "test_ssim",
            "test_lpips",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in curve:
                writer.writerow({key: row.get(key) for key in fields})

    def make_row(pass_index: int, iteration: int, train: dict[str, Any], test: dict[str, Any]):
        return {
            "pass": int(pass_index),
            "refinement_iteration": int(iteration),
            "train_num_views": int(train.get("num_views", 0)),
            "test_num_views": int(test.get("num_views", 0)),
            "train_psnr": train.get("psnr"),
            "train_ssim": train.get("ssim"),
            "train_lpips": train.get("lpips"),
            "test_psnr": test.get("psnr"),
            "test_ssim": test.get("ssim"),
            "test_lpips": test.get("lpips"),
        }

    def print_row(row: dict[str, Any]) -> None:
        print(
            "[global-refine/pass-metrics] "
            f"pass={row['pass']:2d} iter={row['refinement_iteration']:6d} "
            f"Train={row['train_psnr']:.3f}/{row['train_ssim']:.4f}/{row['train_lpips']:.4f} "
            f"Test={row['test_psnr']:.3f}/{row['test_ssim']:.4f}/{row['test_lpips']:.4f}",
            flush=True,
        )

    def finalize_with_pass_refinement(self):
        if (
            self.train_packet_count <= 0
            or self.gaussians is None
            or int(self.gaussians.get_xyz.shape[0]) == 0
        ):
            return original_finalize_impl(self)

        from utils.loss_utils import l1_loss, ssim

        ntrain = len(self.train_cameras)
        ntest = len(self.test_cameras)
        online_iteration_end = int(self.global_iteration)
        g_before = int(self.gaussians.get_xyz.shape[0])

        print(
            "\n[global-refine/pass-metrics] starting: "
            f"train_views={ntrain} test_views={ntest} passes={passes} "
            f"total_refinement_iterations={passes * ntrain} G={g_before}",
            flush=True,
        )

        # Pass 0: exact paired Online-only endpoint, before opacity reset.
        online_train = self._evaluate(self.train_cameras)
        online_test = self._evaluate(self.test_cameras)
        curve = [make_row(0, 0, online_train, online_test)]
        print_row(curve[-1])
        write_curve(self, curve)

        # Use the native GraphDECO reset so the opacity parameter is replaced
        # optimizer-safely while xyz/SH/scale/rotation Adam state is retained.
        if not hasattr(self.gaussians, "reset_opacity"):
            raise RuntimeError("GaussianModel has no reset_opacity()")
        opacity_before = self.gaussians.get_opacity.detach().clone()
        self.gaussians.reset_opacity()
        opacity_after = self.gaussians.get_opacity.detach()
        reset_stats = {
            "enabled": True,
            "implementation": "GaussianModel.reset_opacity()",
            "before_mean": float(opacity_before.mean().item()),
            "before_max": float(opacity_before.max().item()),
            "after_mean": float(opacity_after.mean().item()),
            "after_max": float(opacity_after.max().item()),
        }
        print("[global-refine/pass-metrics] opacity reset:", reset_stats, flush=True)

        # Continue the online optimizer. Do NOT call training_setup().
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        rng = random.Random(seed)
        refine_iteration = 0

        for pass_index in range(1, passes + 1):
            order = list(range(ntrain))
            rng.shuffle(order)

            for camera_index in order:
                refine_iteration += 1
                self.global_iteration += 1
                # Keep online Adam state, while using a refinement-local xyz LR index.
                self.gaussians.update_learning_rate(refine_iteration)

                camera = self.train_cameras[camera_index]
                background = (
                    torch.rand(3, device=self.device)
                    if self.opt.random_background
                    else self.background
                )
                pkg = self.render(
                    camera,
                    self.gaussians,
                    self.pipe,
                    background,
                    use_trained_exp=False,
                    separate_sh=False,
                )
                image = pkg["render"]
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

            # Evaluation is aligned with the budget unit: exactly once per full pass.
            train_metrics = self._evaluate(self.train_cameras)
            test_metrics = self._evaluate(self.test_cameras)
            row = make_row(pass_index, refine_iteration, train_metrics, test_metrics)
            curve.append(row)
            print_row(row)
            write_curve(self, curve)

        # Keep the normal final backend summary, now using the LPIPS-aware evaluator.
        summary = original_finalize_impl(self)
        refine_summary = {
            "enabled": True,
            "protocol": "20-pass post-hoc global refinement with per-pass PSNR/SSIM/LPIPS evaluation",
            "online_iteration_end": online_iteration_end,
            "num_train_views": ntrain,
            "num_test_views": ntest,
            "passes": passes,
            "total_refinement_iterations": passes * ntrain,
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
            "metrics": ["psnr", "ssim", "lpips_vgg"],
            "lpips_convention": "lpips.LPIPS(net='vgg'), normalize=True, input range [0,1]",
            "curve_json": str(self.output_dir / "posthoc_global_refinement_pass_metrics.json"),
            "curve_csv": str(self.output_dir / "posthoc_global_refinement_pass_metrics.csv"),
            "online_before_refinement": curve[0],
            "final_after_refinement": curve[-1],
        }
        summary["posthoc_global_refinement"] = refine_summary
        self._save_json("incremental_backend_summary.json", summary)
        self._save_json("posthoc_global_refinement_summary.json", refine_summary)
        return summary

    BackendEvaluationMixin._finalize_impl = finalize_with_pass_refinement


if __name__ == "__main__":
    _disable_intermediate_online_evaluation()
    _install_lpips_evaluator()
    _install_pass_refinement()
    safe.repro.main()
