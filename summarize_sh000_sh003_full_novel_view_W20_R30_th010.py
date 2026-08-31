#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SEQS = ["SH000", "SH001", "SH002", "SH003"]
EMPTY_RE = re.compile(r"\[empty-map guard\].*?frame=(?P<frame>\d+)")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fnum(v: Any, digits: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{digits}f}"


def empty_frames(path: Path) -> list[int]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [int(m.group("frame")) for m in EMPTY_RE.finditer(text)]


def nearest_g(timing: list[dict[str, Any]], target: int) -> int | None:
    rows = [r for r in timing if r.get("frame_index") is not None and int(r["frame_index"]) <= target]
    if not rows:
        return None
    row = max(rows, key=lambda r: int(r["frame_index"]))
    return int(row["num_gaussians"])


def mean(xs: list[float]) -> float | None:
    return None if not xs else sum(xs) / len(xs)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for seq in SEQS:
        name = f"{seq}_full_novelview_8to2_W20_R30_th010"
        work = Path("outputs") / name
        summary_path = work / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            status_path = work / "exit_status.txt"
            status = f"failed({status_path.read_text().strip()})" if status_path.is_file() else "missing"
            rows.append({"sequence": seq, "status": status, "work_dir": str(work)})
            continue

        s = load_json(summary_path)
        b = s.get("backend") or {}
        m = b.get("final_metrics") or {}
        train = m.get("train_inserted") or {}
        test = m.get("test_all") or {}
        active = m.get("active_local_map") or {}
        tp, xp = train.get("psnr"), test.get("psnr")

        output_name = f"incremental_{name}"
        timing_path = work / output_name / "timing_log.json"
        timing = load_json(timing_path) if timing_path.is_file() else []
        backend_times = [float(r["backend_total_sec"]) for r in timing if r.get("backend_total_sec") is not None]
        events = empty_frames(work / "run.log")

        rows.append({
            "sequence": seq,
            "status": "complete",
            "train_views": train.get("num_views"),
            "test_views": test.get("num_views"),
            "train_psnr": tp,
            "test_psnr": xp,
            "gap_db": None if tp is None or xp is None else float(tp) - float(xp),
            "train_ssim": train.get("ssim"),
            "test_ssim": test.get("ssim"),
            "active_psnr": active.get("psnr"),
            "final_gaussians": b.get("final_num_gaussians"),
            "wall_sec": s.get("streaming_wall_time_sec"),
            "backend_mean_sec": mean(backend_times),
            "backend_last50_mean_sec": mean(backend_times[-50:]),
            "g_at_200": nearest_g(timing, 199),
            "g_at_400": nearest_g(timing, 399),
            "empty_frames": events,
            "ate_se3_rmse_m": nested(s.get("pose_metrics") or {}, "se3", "translation_error_m", "rmse"),
            "work_dir": str(work),
        })

    print("\n=== Full SH000-SH003 strict 8:2 | W20 R30 Th=.10 ===\n")
    header = (
        f"{'Seq':<6} {'Status':<10} {'Train':>5} {'Test':>5} {'TrainP':>8} {'TestP':>8} "
        f"{'Gap':>7} {'TrainS':>8} {'TestS':>8} {'G(M)':>8} {'Wall(s)':>9} {'Last50':>8} {'Empty':>5} {'ATE':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] != "complete":
            print(f"{r['sequence']:<6} {r['status']:<10}")
            continue
        g = r.get("final_gaussians")
        print(
            f"{r['sequence']:<6} {'complete':<10} {str(r.get('train_views')):>5} {str(r.get('test_views')):>5} "
            f"{fnum(r.get('train_psnr')):>8} {fnum(r.get('test_psnr')):>8} {fnum(r.get('gap_db')):>7} "
            f"{fnum(r.get('train_ssim'),4):>8} {fnum(r.get('test_ssim'),4):>8} "
            f"{('-' if g is None else f'{int(g)/1e6:.3f}'):>8} {fnum(r.get('wall_sec'),1):>9} "
            f"{fnum(r.get('backend_last50_mean_sec'),3):>8} {len(r.get('empty_frames', [])):>5} "
            f"{fnum(r.get('ate_se3_rmse_m'),4):>7}"
        )

    print("\n=== Gaussian growth ===\n")
    print(f"{'Seq':<6} {'G@200(M)':>10} {'G@400(M)':>10} {'G@end(M)':>10} {'BackendMean':>12}")
    print("-" * 54)
    for r in rows:
        if r["status"] != "complete":
            continue
        def gm(v: Any) -> str:
            return "-" if v is None else f"{int(v)/1e6:.3f}"
        print(
            f"{r['sequence']:<6} {gm(r.get('g_at_200')):>10} {gm(r.get('g_at_400')):>10} "
            f"{gm(r.get('final_gaussians')):>10} {fnum(r.get('backend_mean_sec'),3):>12}"
        )
        if r.get("empty_frames"):
            print(f"  empty-map frames: {r['empty_frames']}")

    complete = [r for r in rows if r.get("status") == "complete"]
    if complete:
        tests = [float(r["test_psnr"]) for r in complete if r.get("test_psnr") is not None]
        if tests:
            print(f"\nMean Test PSNR over completed sequences: {sum(tests)/len(tests):.3f} dB")

    out = Path("outputs/SH000_SH003_full_novelview_8to2_W20_R30_th010_summary.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
