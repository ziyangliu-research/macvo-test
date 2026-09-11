#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

ETH3D_ROOT = Path(
    "/home/shiyo/Desktop/MAC-VO/outputs/eth3d4_native3dgs_refine100k"
)
TARTANAIR_ROOT = Path(
    "/home/shiyo/Desktop/MAC-VO/outputs/tartanair_se002_sh003_native3dgs_refine100k"
)

ETH3D_SEQS = (
    "mannequin_face_1",
    "einstein_1",
    "sofa_3",
    "plant_scene_3",
)
TARTANAIR_SEQS = ("SH000", "SH001", "SH002", "SH003")
TOP_K = 10


def compute_psnr(gt_path: Path, render_path: Path) -> float:
    gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float64)
    render = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float64)
    if gt.shape != render.shape:
        raise ValueError(f"shape mismatch GT={gt.shape} render={render.shape}")
    mse = float(np.mean((gt - render) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def find_render_root(sequence_root: Path) -> Path | None:
    candidates = sorted(sequence_root.glob("quality_seed0/*/online_test_renders"))
    if not candidates:
        candidates = sorted(sequence_root.glob("**/online_test_renders"))
    return candidates[0] if candidates else None


def rank_sequence(dataset: str, sequence: str, sequence_root: Path):
    render_root = find_render_root(sequence_root)
    if render_root is None:
        print(f"\n=== {dataset} / {sequence} ===")
        print("MISSING: online_test_renders not found")
        return [], []

    results = []
    skipped = []
    for folder in render_root.iterdir():
        if not folder.is_dir():
            continue
        gt_path = folder / "gt.png"
        render_path = folder / "render.png"
        if not gt_path.is_file() or not render_path.is_file():
            skipped.append((folder.name, "missing gt.png or render.png"))
            continue
        try:
            psnr = compute_psnr(gt_path, render_path)
            try:
                frame_index = int(folder.name)
            except ValueError:
                frame_index = -1
            results.append((psnr, frame_index, folder.name, str(folder)))
        except Exception as exc:
            skipped.append((folder.name, str(exc)))

    results.sort(key=lambda x: (-x[0], x[1], x[2]))

    print(f"\n=== {dataset} / {sequence} ===")
    print(f"Render root: {render_root}")
    print(f"Evaluated: {len(results)} views")
    if skipped:
        print(f"Skipped  : {len(skipped)} views")
    print(f"{'Rank':>4}  {'Folder':>12}  {'PSNR (dB)':>12}")
    print("-" * 34)
    for rank, (psnr, _frame_index, folder_name, _folder_path) in enumerate(
        results[:TOP_K], start=1
    ):
        psnr_str = "inf" if math.isinf(psnr) else f"{psnr:.4f}"
        print(f"{rank:>4}  {folder_name:>12}  {psnr_str:>12}")

    return results, skipped


def main() -> None:
    all_top_rows = []
    all_rows = []

    jobs = [
        *(('ETH3D', seq, ETH3D_ROOT / seq) for seq in ETH3D_SEQS),
        *(('TartanAir', seq, TARTANAIR_ROOT / seq) for seq in TARTANAIR_SEQS),
    ]

    for dataset, sequence, sequence_root in jobs:
        results, _ = rank_sequence(dataset, sequence, sequence_root)
        for rank, (psnr, frame_index, folder_name, folder_path) in enumerate(
            results, start=1
        ):
            row = {
                "dataset": dataset,
                "sequence": sequence,
                "rank": rank,
                "frame_index": frame_index if frame_index >= 0 else folder_name,
                "folder": folder_name,
                "psnr_db": psnr,
                "path": folder_path,
            }
            all_rows.append(row)
            if rank <= TOP_K:
                all_top_rows.append(row)

    out_dir = Path("/home/shiyo/Desktop/MAC-VO/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    top_csv = out_dir / "online_test_psnr_top10.csv"
    all_csv = out_dir / "online_test_psnr_all.csv"
    fieldnames = ["dataset", "sequence", "rank", "frame_index", "folder", "psnr_db", "path"]

    for path, rows in ((top_csv, all_top_rows), (all_csv, all_rows)):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\n============================================================")
    print(f"Top-10 CSV: {top_csv}")
    print(f"All-view CSV: {all_csv}")
    print("============================================================")


if __name__ == "__main__":
    main()
