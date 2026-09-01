#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEQS = ["SH000", "SH001", "SH002", "SH003"]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fps_without_metric_render(work_dir: Path, num_frames: int) -> float | None:
    p = work_dir / "frame_timing_log.json"
    if not p.is_file():
        return None
    rows = load(p)
    ends = [float(r["backend_end_sec"]) for r in rows if r.get("backend_end_sec") is not None]
    if not ends:
        return None
    elapsed = max(ends)
    return None if elapsed <= 0 else float(num_frames) / elapsed


def fmt(v: Any, digits: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []
    for seq in SEQS:
        work_dir = Path(f"outputs/{seq}_full_final_W20_R30_B100_th010")
        p = work_dir / "execution_benchmark_summary.json"
        if not p.is_file():
            rows.append({"sequence": seq, "status": "missing"})
            continue

        s = load(p)
        b = s.get("backend") or {}
        fm = b.get("final_metrics") or {}
        train = fm.get("train_inserted") or {}
        test = fm.get("test_all") or {}
        n = int(s.get("num_frames") or 0)
        updates = int(s.get("num_backend_updates") or 0)
        max_map = None if n <= 0 else 100.0 * updates / n

        rows.append({
            "sequence": seq,
            "status": "complete",
            "max_map_pct": max_map,
            "train_psnr": train.get("psnr"),
            "train_ssim": train.get("ssim"),
            "test_psnr": test.get("psnr"),
            "test_ssim": test.get("ssim"),
            "ate_m": nested(s.get("pose_metrics") or {}, "se3", "translation_error_m", "rmse"),
            "fps": fps_without_metric_render(work_dir, n),
            "gaussians": b.get("final_num_gaussians"),
            "num_frames": n,
            "num_backend_updates": updates,
        })

    print("\n=== Final SH000-SH003 results | W20 R30 B100 M50 Th=.10 ===\n")
    header = (
        f"{'Sequence':<8} {'MaxMap':>8} {'Train PSNR/SSIM':>20} "
        f"{'Test PSNR/SSIM':>20} {'ATE(m)':>9} {'FPS':>8} {'Gaussians':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] != "complete":
            print(f"{r['sequence']:<8} {'missing':>8}")
            continue
        g = r.get("gaussians")
        train_pair = f"{fmt(r.get('train_psnr'))}/{fmt(r.get('train_ssim'),4)}"
        test_pair = f"{fmt(r.get('test_psnr'))}/{fmt(r.get('test_ssim'),4)}"
        print(
            f"{r['sequence']:<8} "
            f"{fmt(r.get('max_map_pct'),2)+'%':>8} "
            f"{train_pair:>20} "
            f"{test_pair:>20} "
            f"{fmt(r.get('ate_m'),4):>9} "
            f"{fmt(r.get('fps'),4):>8} "
            f"{('-' if g is None else str(int(g))):>12}"
        )

    out = Path("outputs/SH000_SH003_full_final_W20_R30_B100_th010_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("FPS definition: num_frames / last backend_end_sec; initialization and final metric rendering are excluded.")


if __name__ == "__main__":
    main()
