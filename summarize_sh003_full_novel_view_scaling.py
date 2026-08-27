#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CASES = [
    {
        "key": "v0",
        "label": "V0 r0 th.005",
        "name": "v0_baseline_r0_th0005",
        "replay": 0.0,
        "threshold": 0.005,
    },
    {
        "key": "v1",
        "label": "V1 r30 th.005",
        "name": "v1_replay30_th0005",
        "replay": 0.30,
        "threshold": 0.005,
    },
    {
        "key": "v2",
        "label": "V2 r30 th.05",
        "name": "v2_replay30_th005",
        "replay": 0.30,
        "threshold": 0.05,
    },
    {
        "key": "v3",
        "label": "V3 r30 th.10",
        "name": "v3_replay30_th010",
        "replay": 0.30,
        "threshold": 0.10,
    },
]

EMPTY_RE = re.compile(
    r"\[empty-map guard\]\s+packet=(?P<packet>\d+)\s+"
    r"frame=(?P<frame>\d+)\s+first_skipped_local_iter=(?P<iter>\d+)"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def nearest_growth(timing: list[dict[str, Any]], target_frame: int) -> int | None:
    eligible = [
        row
        for row in timing
        if row.get("frame_index") is not None
        and int(row["frame_index"]) <= target_frame
        and row.get("num_gaussians") is not None
    ]
    if not eligible:
        return None
    row = max(eligible, key=lambda x: int(x["frame_index"]))
    return int(row["num_gaussians"])


def empty_events(log_path: Path) -> list[dict[str, int]]:
    if not log_path.is_file():
        return []
    events: list[dict[str, int]] = []
    for match in EMPTY_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        events.append(
            {
                "packet": int(match.group("packet")),
                "frame": int(match.group("frame")),
                "first_skipped_local_iter": int(match.group("iter")),
            }
        )
    return events


def fnum(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def g_m(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1e6:.3f}"


def main() -> None:
    rows: list[dict[str, Any]] = []

    for case in CASES:
        work_dir = Path(f"outputs/SH003_full_novelview_8to2_{case['name']}")
        summary_path = work_dir / "execution_benchmark_summary.json"
        exit_path = work_dir / "exit_status.txt"
        run_log = work_dir / "run.log"

        if not summary_path.is_file():
            rows.append(
                {
                    **case,
                    "status": (
                        f"failed({exit_path.read_text().strip()})"
                        if exit_path.is_file()
                        else "missing"
                    ),
                    "work_dir": str(work_dir),
                    "empty_events": empty_events(run_log),
                }
            )
            continue

        summary = load_json(summary_path)
        backend = summary.get("backend") or {}
        metrics = backend.get("final_metrics") or {}
        train = metrics.get("train_inserted") or {}
        test = metrics.get("test_all") or {}
        active = metrics.get("active_local_map") or {}
        train_psnr = train.get("psnr")
        test_psnr = test.get("psnr")
        pose = summary.get("pose_metrics") or {}

        output_name = f"incremental_SH003_full_novelview_8to2_{case['name']}"
        timing_path = work_dir / output_name / "timing_log.json"
        timing = load_json(timing_path) if timing_path.is_file() else []

        backend_times = [
            float(row["backend_total_sec"])
            for row in timing
            if row.get("backend_total_sec") is not None
        ]
        tail_backend_times = backend_times[-50:]
        final_frame = max(
            [int(row["frame_index"]) for row in timing if row.get("frame_index") is not None],
            default=None,
        )

        events = empty_events(run_log)
        row = {
            **case,
            "status": "complete",
            "work_dir": str(work_dir),
            "num_frames": summary.get("num_frames"),
            "train_views": train.get("num_views"),
            "test_views": test.get("num_views"),
            "train_psnr": train_psnr,
            "test_psnr": test_psnr,
            "gap_db": (
                None
                if train_psnr is None or test_psnr is None
                else float(train_psnr) - float(test_psnr)
            ),
            "active_psnr": active.get("psnr"),
            "train_ssim": train.get("ssim"),
            "test_ssim": test.get("ssim"),
            "final_gaussians": backend.get("final_num_gaussians"),
            "wall_sec": summary.get("streaming_wall_time_sec"),
            "quality_protocol_fps": summary.get("streaming_fps_excluding_initialization"),
            "ate_se3_rmse_m": nested(pose, "se3", "translation_error_m", "rmse"),
            "empty_events": events,
            "empty_event_count": len(events),
            "empty_frames": [event["frame"] for event in events],
            "timing_num_packets": len(timing),
            "backend_mean_sec": mean(backend_times),
            "backend_last50_mean_sec": mean(tail_backend_times),
            "last_timing_frame": final_frame,
            "g_at_frame_200": nearest_growth(timing, 199),
            "g_at_frame_400": nearest_growth(timing, 399),
            "g_at_frame_800": nearest_growth(timing, 799),
            "g_at_frame_1200": nearest_growth(timing, 1199),
        }
        rows.append(row)

    print("\n=== Full SH003 strict 8:2 novel-view scaling ===\n")
    header = (
        f"{'System':<17} {'Status':<10} {'Train':>5} {'Test':>5} "
        f"{'TrainP':>7} {'TestP':>7} {'Gap':>6} {'Active':>7} "
        f"{'TestSSIM':>8} {'G(M)':>8} {'Wall(s)':>9} {'Empty':>5} {'ATE':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["status"] != "complete":
            print(
                f"{row['label']:<17} {row['status']:<10} "
                f"{'-':>5} {'-':>5} {'-':>7} {'-':>7} {'-':>6} {'-':>7} "
                f"{'-':>8} {'-':>8} {'-':>9} {len(row.get('empty_events', [])):>5} {'-':>7}"
            )
            continue
        g = row.get("final_gaussians")
        print(
            f"{row['label']:<17} {'complete':<10} "
            f"{str(row.get('train_views')):>5} {str(row.get('test_views')):>5} "
            f"{fnum(row.get('train_psnr')):>7} {fnum(row.get('test_psnr')):>7} "
            f"{fnum(row.get('gap_db')):>6} {fnum(row.get('active_psnr')):>7} "
            f"{fnum(row.get('test_ssim'),4):>8} "
            f"{('-' if g is None else f'{int(g)/1e6:.3f}'):>8} "
            f"{fnum(row.get('wall_sec'),1):>9} "
            f"{int(row.get('empty_event_count',0)):>5} "
            f"{fnum(row.get('ate_se3_rmse_m'),4):>7}"
        )

    print("\n=== Gaussian growth / backend scaling ===\n")
    header2 = (
        f"{'System':<17} {'G@200M':>8} {'G@400M':>8} {'G@800M':>8} "
        f"{'G@1200M':>9} {'BackendMean':>12} {'Last50Mean':>11}"
    )
    print(header2)
    print("-" * len(header2))
    for row in rows:
        if row.get("status") != "complete":
            continue
        print(
            f"{row['label']:<17} "
            f"{g_m(row.get('g_at_frame_200')):>8} "
            f"{g_m(row.get('g_at_frame_400')):>8} "
            f"{g_m(row.get('g_at_frame_800')):>8} "
            f"{g_m(row.get('g_at_frame_1200')):>9} "
            f"{fnum(row.get('backend_mean_sec'),3):>12} "
            f"{fnum(row.get('backend_last50_mean_sec'),3):>11}"
        )
        if row.get("empty_events"):
            print(f"  empty-map frames: {row['empty_frames']}")

    out = Path("outputs/SH003_full_novelview_8to2_scaling_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
