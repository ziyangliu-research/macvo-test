#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def fnum(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    if len(sys.argv) > 1:
        work_dir = Path(sys.argv[1])
    else:
        work_dir = Path(
            "outputs/SH003_0_200_final_W20_R30_B100_th010_global_refine5x"
        )

    curve_path = work_dir / "incremental_SH003_0_200_final_W20_R30_B100_th010_global_refine5x" / "posthoc_global_refinement_curve.json"
    summary_path = work_dir / "execution_benchmark_summary.json"

    # Backend artifacts live under backend.output_name.  Fall back to discovery
    # so the summarizer also works if the output name is changed later.
    if not curve_path.is_file():
        matches = list(work_dir.glob("*/posthoc_global_refinement_curve.json"))
        if len(matches) == 1:
            curve_path = matches[0]
        elif not matches:
            raise FileNotFoundError(
                f"posthoc_global_refinement_curve.json not found below {work_dir}"
            )
        else:
            raise RuntimeError(f"multiple refinement curves found below {work_dir}: {matches}")

    curve = json.loads(curve_path.read_text(encoding="utf-8"))

    print("\n=== SH003 Post-hoc Global Refinement Curve ===")
    print("Online final W20/R30/B100/Th=.10 -> global opacity reset -> shuffled all-train refinement")
    print("Optimization speed excludes Train/Test metric rendering.\n")

    header = (
        f"{'Iter':>5} {'Pass':>6} {'TrainP':>8} {'TrainS':>8} "
        f"{'TestP':>8} {'TestS':>8} {'G(k)':>9} {'Opt(s)':>9} {'Upd/s':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in curve:
        g = row.get("num_gaussians")
        gk = None if g is None else int(g) / 1000.0
        print(
            f"{int(row.get('refinement_iteration', 0)):>5} "
            f"{fnum(row.get('equivalent_passes'),2):>6} "
            f"{fnum(row.get('train_psnr')):>8} "
            f"{fnum(row.get('train_ssim'),4):>8} "
            f"{fnum(row.get('test_psnr')):>8} "
            f"{fnum(row.get('test_ssim'),4):>8} "
            f"{fnum(gk,1):>9} "
            f"{fnum(row.get('cumulative_optimization_sec'),1):>9} "
            f"{fnum(row.get('optimization_view_updates_per_sec'),3):>8}"
        )

    valid = [r for r in curve if r.get("test_psnr") is not None]
    refined = [r for r in valid if int(r.get("refinement_iteration", 0)) > 0]
    if valid:
        baseline = valid[0]
        print(
            "\nOnline-only paired baseline: "
            f"Train {float(baseline['train_psnr']):.3f}/{float(baseline['train_ssim']):.4f}, "
            f"Test {float(baseline['test_psnr']):.3f}/{float(baseline['test_ssim']):.4f}"
        )
    if refined:
        best = max(refined, key=lambda x: float(x["test_psnr"]))
        print(
            "Best held-out Test PSNR: "
            f"iter={best['refinement_iteration']} "
            f"({float(best['equivalent_passes']):.2f} passes), "
            f"PSNR={float(best['test_psnr']):.3f} dB, "
            f"SSIM={float(best['test_ssim']):.4f}"
        )

    if summary_path.is_file():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        refine = (s.get("backend") or {}).get("posthoc_global_refinement") or {}
        speed = refine.get("optimization_view_updates_per_sec")
        opt_sec = refine.get("optimization_only_sec")
        if speed is not None:
            print(
                "Global-refinement optimization-only speed: "
                f"{float(speed):.3f} train-view updates/s "
                "(FPS-like throughput; not online pipeline FPS)"
            )
        if opt_sec is not None:
            print(f"Global-refinement optimization-only time: {float(opt_sec):.1f} s")

    print(f"\nCurve JSON: {curve_path}")
    print(f"Curve CSV : {curve_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
