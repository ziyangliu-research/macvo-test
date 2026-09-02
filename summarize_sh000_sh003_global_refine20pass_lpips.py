#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

SEQS = ["SH000", "SH001", "SH002", "SH003"]
ROOT = Path("outputs")


def f(value: Any, digits: int) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def find_curve(seq: str) -> Path:
    work_dir = ROOT / f"{seq}_full_final_W20_R30_B100_th010_global_refine20pass_lpips"
    matches = list(work_dir.glob("*/posthoc_global_refinement_pass_metrics.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one pass-metric curve below {work_dir}, found {matches}"
        )
    return matches[0]


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []

    for seq in SEQS:
        curve_path = find_curve(seq)
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        if not curve:
            raise RuntimeError(f"empty curve: {curve_path}")

        for row in curve:
            out = {"sequence": seq, **row}
            all_rows.append(out)

        online = curve[0]
        final = curve[-1]
        endpoints.append(
            {
                "sequence": seq,
                "online_train_psnr": online.get("train_psnr"),
                "online_train_ssim": online.get("train_ssim"),
                "online_train_lpips": online.get("train_lpips"),
                "online_test_psnr": online.get("test_psnr"),
                "online_test_ssim": online.get("test_ssim"),
                "online_test_lpips": online.get("test_lpips"),
                "pass20_train_psnr": final.get("train_psnr"),
                "pass20_train_ssim": final.get("train_ssim"),
                "pass20_train_lpips": final.get("train_lpips"),
                "pass20_test_psnr": final.get("test_psnr"),
                "pass20_test_ssim": final.get("test_ssim"),
                "pass20_test_lpips": final.get("test_lpips"),
            }
        )

        print(f"\n=== {seq} | per-pass quality ===")
        header = (
            f"{'Pass':>4} {'Iter':>7} | "
            f"{'TrainP':>7} {'TrainS':>7} {'TrainL':>7} | "
            f"{'TestP':>7} {'TestS':>7} {'TestL':>7}"
        )
        print(header)
        print("-" * len(header))
        for row in curve:
            print(
                f"{int(row['pass']):>4} {int(row['refinement_iteration']):>7} | "
                f"{f(row.get('train_psnr'),3):>7} "
                f"{f(row.get('train_ssim'),4):>7} "
                f"{f(row.get('train_lpips'),4):>7} | "
                f"{f(row.get('test_psnr'),3):>7} "
                f"{f(row.get('test_ssim'),4):>7} "
                f"{f(row.get('test_lpips'),4):>7}"
            )

    long_path = ROOT / "SH000_SH003_global_refine20pass_quality_curve_lpips.csv"
    long_fields = [
        "sequence", "pass", "refinement_iteration", "train_num_views", "test_num_views",
        "train_psnr", "train_ssim", "train_lpips",
        "test_psnr", "test_ssim", "test_lpips",
    ]
    with long_path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=long_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key) for key in long_fields})

    endpoint_path = ROOT / "SH000_SH003_global_refine20pass_online_vs_final_lpips.csv"
    endpoint_fields = list(endpoints[0].keys())
    with endpoint_path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=endpoint_fields)
        writer.writeheader()
        writer.writerows(endpoints)

    print("\n=== Online vs 20-pass Global Refinement ===")
    header = (
        f"{'Seq':<6} | {'Online Train P/S/L':>25} | {'Online Test P/S/L':>25} | "
        f"{'Pass20 Train P/S/L':>25} | {'Pass20 Test P/S/L':>25}"
    )
    print(header)
    print("-" * len(header))
    for row in endpoints:
        print(
            f"{row['sequence']:<6} | "
            f"{f(row['online_train_psnr'],3)}/{f(row['online_train_ssim'],4)}/{f(row['online_train_lpips'],4):<7} | "
            f"{f(row['online_test_psnr'],3)}/{f(row['online_test_ssim'],4)}/{f(row['online_test_lpips'],4):<7} | "
            f"{f(row['pass20_train_psnr'],3)}/{f(row['pass20_train_ssim'],4)}/{f(row['pass20_train_lpips'],4):<7} | "
            f"{f(row['pass20_test_psnr'],3)}/{f(row['pass20_test_ssim'],4)}/{f(row['pass20_test_lpips'],4):<7}"
        )

    print(f"\nSaved full curves : {long_path}")
    print(f"Saved endpoints   : {endpoint_path}")


if __name__ == "__main__":
    main()
