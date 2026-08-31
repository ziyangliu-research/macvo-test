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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fnum(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def empty_frames(path: Path) -> list[int]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [int(m.group("frame")) for m in EMPTY_RE.finditer(text)]


def main() -> None:
    rows: list[dict[str, Any]] = []

    for seq in SEQS:
        name = f"{seq}_full_timing_alltrain_W20_R30_B100_th010"
        work_dir = Path("outputs") / name
        summary_path = work_dir / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            rows.append({"seq": seq, "status": "missing", "work_dir": str(work_dir)})
            continue

        s = load_json(summary_path)
        backend = s.get("backend") or {}
        output_name = f"incremental_{name}"

        frame_timing_path = work_dir / "frame_timing_log.json"
        frame_timing = load_json(frame_timing_path) if frame_timing_path.is_file() else []
        pose_times = [
            float(x["pose_task_duration_sec"])
            for x in frame_timing
            if x.get("pose_task_duration_sec") is not None
        ]
        packet_times = [
            float(x["packet_duration_sec"])
            for x in frame_timing
            if x.get("packet_duration_sec") is not None
        ]
        backend_times = [
            float(x["backend_duration_sec"])
            for x in frame_timing
            if x.get("backend_duration_sec") is not None
        ]

        backend_timing_path = work_dir / output_name / "timing_log.json"
        backend_timing = load_json(backend_timing_path) if backend_timing_path.is_file() else []
        peak_vram_values: list[float] = []
        for x in backend_timing:
            for key in (
                "gpu_peak_memory_allocated_gb",
                "gpu_memory_allocated_gb",
                "gpu_used_global_gb",
            ):
                if x.get(key) is not None:
                    peak_vram_values.append(float(x[key]))
                    break

        num_frames = int(s.get("num_frames") or 0)
        wall = float(s.get("streaming_wall_time_sec") or 0.0)
        fps = s.get("streaming_fps_excluding_initialization")
        if fps is None and wall > 0:
            fps = num_frames / wall

        rows.append(
            {
                "seq": seq,
                "status": "complete",
                "frames": num_frames,
                "train_frames": s.get("num_train_frames"),
                "test_frames": s.get("num_test_frames"),
                "wall_sec": wall,
                "fps": fps,
                "initialization_sec": s.get("initialization_sec"),
                "pose_mean_ms": None if not pose_times else 1000.0 * mean(pose_times),
                "resplat_mean_ms": None if not packet_times else 1000.0 * mean(packet_times),
                "backend_mean_ms": None if not backend_times else 1000.0 * mean(backend_times),
                "backend_last50_ms": None if not backend_times else 1000.0 * mean(backend_times[-50:]),
                "final_gaussians": backend.get("final_num_gaussians"),
                "peak_vram_gb": max(peak_vram_values) if peak_vram_values else None,
                "empty_frames": empty_frames(work_dir / "run.log"),
                "work_dir": str(work_dir),
            }
        )

    print("\n=== SH000-SH003 formal all-train timing | W20 rho=.30 B100 M50 Th=.10 ===\n")
    header = (
        f"{'Seq':<6} {'Status':<9} {'Frames':>6} {'Wall(s)':>9} {'FPS':>7} "
        f"{'Pose(ms)':>9} {'ReSplat(ms)':>11} {'Backend(ms)':>11} {'Last50(ms)':>11} "
        f"{'G(M)':>8} {'PeakGB':>8} {'Empty':>5}"
    )
    print(header)
    print("-" * len(header))

    complete: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") != "complete":
            print(f"{r['seq']:<6} {'missing':<9}")
            continue
        complete.append(r)
        g = r.get("final_gaussians")
        gm = None if g is None else int(g) / 1e6
        print(
            f"{r['seq']:<6} {'complete':<9} {int(r['frames']):>6} "
            f"{fnum(r.get('wall_sec'),1):>9} {fnum(r.get('fps'),3):>7} "
            f"{fnum(r.get('pose_mean_ms'),1):>9} {fnum(r.get('resplat_mean_ms'),1):>11} "
            f"{fnum(r.get('backend_mean_ms'),1):>11} {fnum(r.get('backend_last50_ms'),1):>11} "
            f"{fnum(gm,3):>8} {fnum(r.get('peak_vram_gb'),2):>8} "
            f"{len(r.get('empty_frames', [])):>5}"
        )
        if r.get("empty_frames"):
            print(f"  empty-map frames: {r['empty_frames']}")

    if complete:
        total_frames = sum(int(r["frames"]) for r in complete)
        total_wall = sum(float(r["wall_sec"]) for r in complete)
        weighted_fps = total_frames / total_wall if total_wall > 0 else None
        mean_fps = mean([float(r["fps"]) for r in complete if r.get("fps") is not None])
        print("\nAggregate:")
        print(f"  total frames       = {total_frames}")
        print(f"  total streaming s  = {total_wall:.1f}")
        print(f"  weighted FPS       = {fnum(weighted_fps,3)}")
        print(f"  mean sequence FPS  = {fnum(mean_fps,3)}")
        print("\nFPS definition: num_frames / streaming_wall_time_sec (model initialization excluded).")
        print("All frames are mapping frames; backend evaluation and final PLY saving are disabled.")

    out = Path("outputs/SH000_SH003_full_timing_alltrain_W20_R30_B100_th010_summary.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
