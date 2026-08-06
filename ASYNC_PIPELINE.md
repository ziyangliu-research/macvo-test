# Asynchronous MAC-VO + ReSplat + Incremental 3DGS

This branch replaces command-to-command subprocess chaining with persistent
Python components and bounded in-memory queues. The sibling repository layout is
kept unchanged:

```text
Desktop/
├── MAC-VO/               # this repository and system entrypoint
├── Resplat/              # ReSplat model code/checkpoints
└── gaussian-splatting/   # GraphDECO renderer/model
```

The serial pipeline remains untouched. Development is isolated on:

```text
feature/async-macvo-resplat-3dgs-v1
```

## Runtime architecture

```text
                         ┌──────────────────────────────┐
Decoded StereoFrame ─────┤ MAC-VO pose worker           │
       │                 │ committed metric Twc (t-1)  │
       │                 └──────────────┬───────────────┘
       │                                │
       │                 ┌──────────────▼───────────────┐
       └────────────────►│ ReSplat packet worker        │
                         │ [I, fixed stereo baseline]    │
                         │ left-camera-local Gaussians   │
                         └──────────────┬───────────────┘
                                        │
                              ordered pose/packet join
                                        │
                         ┌──────────────▼───────────────┐
                         │ Incremental GraphDECO backend │
                         │ local packet -> world frame   │
                         │ reset / append / recent-K opt │
                         └──────────────────────────────┘
```

The models are initialized once, sequentially, in their owning worker threads.
Streaming then uses three bounded workers. Queue backpressure limits outstanding
GPU packets.

## Local-to-world packet contract

ReSplat receives a canonical stereo rig:

```text
left  c2w = I
right c2w = fixed 0.25000006 m stereo transform
```

After MAC-VO commits `T_world_from_left`, the backend applies:

```text
mean_world = R_world_left @ mean_local + t_world_left
cov_world  = R_world_left @ cov_local @ R_world_left^T
```

ReSplat quaternion fields are `xyzw`; GraphDECO uses `wxyz`. The conversion is
explicit.

The current ReSplat configuration uses `no_rotate_sh=true`. A packet inferred in
the canonical frame therefore carries higher-order SH coefficients in the local
left-camera basis. To preserve the appearance of the same packet after a rigid
world transform, the async path rotates SH coefficients with ReSplat's own
Wigner-D implementation:

```text
SH_world = WignerD(R_world_left) @ SH_local
```

Scale and opacity are unchanged by the rigid transform.

## Two different validation questions

The original exact-equivalence test compares:

```text
ReSplat([I, baseline]) + explicit alignment
```

against:

```text
ReSplat([Twc, Twc @ baseline])
```

ReSplat is not guaranteed to be exactly SE(3)-equivariant because its point
transformer constructs discrete KNN neighborhoods from world-space points.
Therefore `validate_resplat_local_world_equivalence.py` is diagnostic only; its
`passed=false` does not by itself reject local-first inference.

The production gate is:

```text
validate_resplat_local_alignment_quality.py
```

It checks separately:

1. The same local packet rendered before and after rigid alignment.
2. Local-first packet GT PSNR/SSIM versus the current direct-world path.
3. Exact learned-model equivariance as non-blocking diagnostic information.

The required top-level result is:

```json
{
  "alignment_passed": true,
  "quality_passed": true,
  "safe_for_async": true,
  "passed": true
}
```

`exact_equivariance_passed` may remain false.

## Validation commands

```bash
cd /home/shiyo/Desktop/MAC-VO

git switch feature/async-macvo-resplat-3dgs-v1
git pull origin feature/async-macvo-resplat-3dgs-v1
```

File-path preprocessing:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python validate_resplat_local_alignment_quality.py \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --dataset_root /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000 \
  --pose_npz /home/shiyo/Desktop/MAC-VO/outputs/macvo_stereo_resplat_P000_0_50/macvo_pose/macvo_pose_results.npz \
  --pose_indices 1,10,20,30,40,49 \
  --input_mode file_paths \
  --output outputs/local_alignment_quality_file_paths.json
```

Shared decoded tensors:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python validate_resplat_local_alignment_quality.py \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --dataset_root /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000 \
  --pose_npz /home/shiyo/Desktop/MAC-VO/outputs/macvo_stereo_resplat_P000_0_50/macvo_pose/macvo_pose_results.npz \
  --pose_indices 1,10,20,30,40,49 \
  --input_mode shared_tensors \
  --output outputs/local_alignment_quality_shared_tensors.json
```

## Null-backend ordering test

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_async_pipeline.py \
  --config Config/Pipeline/MACVO_ReSplat_Async_P000_0_50.yaml \
  --set runtime.backend_mode=null \
  --set paths.work_dir=outputs/macvo_resplat_async_null_P000_0_50
```

This validates:

- persistent MAC-VO/ReSplat initialization;
- parallel frontend task submission;
- one-frame-delayed committed MAC-VO poses;
- ordered pose/packet joining;
- strict 40-train / 10-test split;
- no ReSplat packet generation for test frames.

## Full backend

Compatibility-oriented run:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_async_pipeline.py \
  --config Config/Pipeline/MACVO_ReSplat_Async_P000_0_50.yaml \
  --set resplat_frontend.input_mode=file_paths \
  --set resplat_frontend.handoff_mode=pinned_cpu \
  --set paths.work_dir=outputs/macvo_resplat_async_compat_P000_0_50
```

Default in-memory run:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_async_pipeline.py \
  --config Config/Pipeline/MACVO_ReSplat_Async_P000_0_50.yaml
```

Throughput run:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_async_pipeline.py \
  --config Config/Pipeline/MACVO_ReSplat_Async_Fast_P000_0_50.yaml
```

## Current implementation constraints

- No NPZ/PT/JSON file is required as a stage hand-off.
- MAC-VO commits frame `t-1` after processing `t`.
- Frames marked `need_interp` are skipped online by default.
- The online GraphDECO spatial learning-rate scale is explicit because future
  cameras are not available when the backend starts.
- Dedicated CUDA streams permit overlap but do not guarantee a speedup on one
  GPU; wall time, queue wait, VRAM, and update latency must be measured.
- The branch is not ready for merging until alignment validation, null-backend
  ordering, and serial-versus-async metric comparisons pass on the target GPU.
