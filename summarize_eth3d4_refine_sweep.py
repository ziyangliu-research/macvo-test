#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SEQS = ("mannequin_face_1", "einstein_1", "sofa_3", "plant_scene_3")


def load(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def triplet(m):
    if not isinstance(m, dict) or int(m.get("num_views", 0)) <= 0:
        return (None, None, None)
    return (float(m["psnr"]), float(m["ssim"]), float(m["lpips"]))


def online_wall(path: Path) -> float | None:
    rows = load(path)
    if not isinstance(rows, list):
        return None
    values = [float(r["backend_end_sec"]) for r in rows if "backend_end_sec" in r]
    return max(values) if values else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs/eth3d4_refine_sweep"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    for seq in SEQS:
        work = args.root / seq / f"quality_seed{args.seed}"
        name = f"incremental_ETH3D_{seq}_quality_seed{args.seed}"
        metrics = load(work / name / "posthoc_global_refinement_dual_checkpoint_metrics.json")
        summary = load(work / "execution_benchmark_summary.json")
        if not isinstance(metrics, dict) or not isinstance(summary, dict):
            print(f"[MISSING] {seq}")
            continue

        nframes = int(summary.get("num_frames", 0))
        ntrain = int(metrics.get("num_train_views", 0))
        ntest = int(metrics.get("num_test_views", 0))
        wall = online_wall(work / "frame_timing_log.json")
        fps = nframes / wall if wall and wall > 0 else None
        ate = None
        try:
            ate = float(summary["pose_metrics"]["se3"]["translation_error_m"]["rmse"])
        except Exception:
            pass
        gaussians = None
        try:
            gaussians = int(summary["backend"]["final_num_gaussians"])
        except Exception:
            pass

        online = metrics.get("online") or {}
        rows.append({
            "sequence": seq, "kind": "online", "checkpoint": 0,
            "refinement_iterations": 0, "equivalent_passes": 0.0,
            "train": triplet(online.get("train")), "test": triplet(online.get("test")),
            "frames": nframes, "train_views": ntrain, "test_views": ntest,
            "ate_m": ate, "gaussians": gaussians, "online_fps": fps,
        })

        by_iter = metrics.get("checkpoints_by_iteration") or {}
        for key in sorted(by_iter, key=lambda x: int(x)):
            item = by_iter[key]
            it = int(item["refinement_iterations"])
            rows.append({
                "sequence": seq, "kind": "fixed_iterations", "checkpoint": int(key),
                "refinement_iterations": it,
                "equivalent_passes": float(item.get("equivalent_passes", it / max(ntrain, 1))),
                "train": triplet(item.get("train")), "test": triplet(item.get("test")),
                "frames": nframes, "train_views": ntrain, "test_views": ntest,
                "ate_m": ate, "gaussians": gaussians, "online_fps": None,
            })

        by_pass = metrics.get("checkpoints_by_pass") or {}
        for key in sorted(by_pass, key=lambda x: int(x)):
            item = by_pass[key]
            it = int(item["refinement_iterations"])
            rows.append({
                "sequence": seq, "kind": "fixed_passes", "checkpoint": int(key),
                "refinement_iterations": it,
                "equivalent_passes": float(item.get("equivalent_passes", int(key))),
                "train": triplet(item.get("train")), "test": triplet(item.get("test")),
                "frames": nframes, "train_views": ntrain, "test_views": ntest,
                "ate_m": ate, "gaussians": gaussians, "online_fps": None,
            })

    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = args.root / f"summary_seed{args.seed}.csv"
    json_path = args.root / f"summary_seed{args.seed}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sequence", "kind", "checkpoint", "refinement_iterations", "equivalent_passes",
            "train_psnr", "train_ssim", "train_lpips", "test_psnr", "test_ssim", "test_lpips",
            "frames", "train_views", "test_views", "se3_ate_rmse_m", "gaussians", "online_fps",
        ])
        for r in rows:
            w.writerow([
                r["sequence"], r["kind"], r["checkpoint"], r["refinement_iterations"],
                r["equivalent_passes"], *r["train"], *r["test"], r["frames"],
                r["train_views"], r["test_views"], r["ate_m"], r["gaussians"], r["online_fps"],
            ])
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== Fixed-iteration checkpoints (primary comparison candidate) ===")
    print(f"{'Sequence':20s} {'Iter':>7s} {'PassEq':>8s} {'Test PSNR':>10s} {'Test SSIM':>10s} {'Test LPIPS':>11s}")
    for r in rows:
        if r["kind"] != "fixed_iterations":
            continue
        tp, ts, tl = r["test"]
        print(
            f"{r['sequence']:20s} {r['refinement_iterations']:7d} {r['equivalent_passes']:8.2f} "
            f"{tp if tp is not None else float('nan'):10.3f} "
            f"{ts if ts is not None else float('nan'):10.4f} "
            f"{tl if tl is not None else float('nan'):11.4f}"
        )
    print(f"\nCSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
