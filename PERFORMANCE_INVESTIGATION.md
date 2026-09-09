# ComfyUI Performance Investigation Notes

## System Specs

- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM, compute capability 8.6)
- **RAM**: 32GB
- **Storage**: 1TB Seagate Barracuda 7200rpm HDD (ST1000DM010) — 97% full, ~0.23 GB/s sequential read
- **Swap**: 24GB file swap + 15.6GB zram (barely used: pswpin=12, pswpout=1382)
- **OS**: Linux (swappiness=20)
- **Driver**: 580.173.02 (`nvidia-smi`; driver was upgraded in the same window as the torch upgrade)
- **Compute capability**: 8.6 (Ampere)

### GPU driver / CUDA details (as measured)

| Component | Version | Notes |
|-----------|---------|-------|
| NVIDIA driver | 580.173.02 | Upgraded alongside torch — one of the two suspected causes of the regression |
| CUDA driver API (torch) | 13.0 | `torch.version.cuda` |
| CUDA runtime (pip) | 13.0.2 | `cuda-toolkit` |
| cuda-bindings | 13.3.0 | |
| cuDNN (runtime) | 9.15.1.9 | **Upgraded from 9.13.0.50** to clear the Conv3D workaround — see Issue 2 |
| cuBLAS | 13.0.0.19 | |
| cuSPARSELt | 0.8.0 | |
| NCCL | 2.27.7 | |
| NVJITLink | 13.0.39 | |

### Python / torch stack (venv at `venv/`)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.9.1+cu130 | Requires `nvidia-cudnn-cu13==9.13.0.50` per its metadata — ignored when 9.15.1.9 installed (ABI-compatible) |
| torchvision | 0.24.1+cu130 | |
| torchaudio | 2.9.1+cu130 | |
| triton | 3.5.1 | |
| comfy-kitchen | 0.2.31 | Kitchen CUDA ops gated on cu130+ (`comfy/quant_ops.py`) |
| comfy-aimdo | 0.4.13 | Underlies DynamicVRAM (VBAR faulting); disables CUDA ops with `--disable-dynamic-vram` |

### Relevant torch-version gates hit by this stack

- `NVIDIA_MEMORY_CONV_BUG_WORKAROUND` (`comfy/ops.py:101-110`) activates for cuDNN `[91002, 91500)` AND torch 2.9–2.10 — WAS active on 9.13.0 + torch 2.9.1, cleared by cuDNN 9.15.1.
- DynamicVRAM is auto-enabled on NVIDIA with torch >= 2.8 (`main.py:251-253`) — this is the per-prompt overhead (Issue 1).
- `SDPA_BACKEND_PRIORITY = [FLASH, CUDNN, EFFICIENT, MATH]` active on NVIDIA torch 2.9 (`comfy/ops.py:68-73`).
- `ENABLE_PYTORCH_ATTENTION` auto-set on NVIDIA torch 2.x (`comfy/model_management.py:464-467`).
- `--use-pytorch-cross-attention`, `--use-split-cross-attention`, `--use-quad-cross-attention`, `--use-pytorch-splat-attention` override the default attention path if needed.

## Model Files

| Model | File | Size | Format |
|-------|------|------|--------|
| Klein 9B | `diffusion_models/KLEIN/flux2Klein9BKVInt8_v10.safetensors` | 8.8 GB | int8 KV quantized |
| MiniMax H3 | `diffusion_models/minimax_h3_*` | 20-22 GB | int8 quantized variants |

---

## Issue 1: Klein 9B — Slow on Every Run

### Symptom
Every prompt execution took ~175s, with only ~27s spent in actual sampling. The 150s overhead occurred on every run, not just the first.

### Root Cause: DynamicVRAM Per-Prompt Overhead

The `ModelPatcherDynamic` (enabled by default with `comfy-aimdo`) runs the full `load()` path on **every prompt execution** via `partially_load()` at `model_patcher.py:2141`:

```python
def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
    with self.use_ejected(skip_and_inject_on_exit_only=True):
        dirty = self.model.current_weight_patches_uuid is not None and \
                (self.model.current_weight_patches_uuid != self.patches_uuid)
        self.unpatch_model(self.offload_device, unpatch_weights=False)
        self.patch_model(load_weights=False)
        self.load(device_to, dirty=dirty)  # <-- runs every prompt
```

`load()` (`model_patcher.py:1853`) rebuilds the entire per-module setup:
1. `_load_list()` — iterates every module (~500+ for a 9B model)
2. Per-module `setup_param()` + `get_key_weight()` loop
3. VBAR allocation checks (skipped if `_v` exists, but loop still runs)
4. Buffer copies to GPU

Even when `dirty=False` (no LoRA changes, same prompt), the full per-module Python loop runs. The VBAR data stays resident (signatures preserved), but the setup overhead adds up across 500+ modules.

### Why `--disable-dynamic-vram` Fixes It

With `--disable-dynamic-vram`, the legacy `ModelPatcher` is used instead:
- First run: slower (disk→RAM→VRAM, estimation-based partial load)
- Subsequent runs: fast (~52s total, 27s sampling) because weights stay fully resident in VRAM after first load
- No per-module VBAR setup loop on each prompt

### Verdict
For Klein 9B (8.8GB quantized on 12GB VRAM), the legacy patcher is the better choice. The model fits partially in VRAM, and the legacy path keeps it resident without per-prompt rebuild overhead.

**Working config for Klein 9B:**
```
--disable-dynamic-vram --cache-classic
```

### What Was NOT the Bottleneck

1. **Swappiness** — Not the issue. Model stays in RAM (22GB available, 8.8GB model). Swap barely used.
2. **Disk I/O** — 0.23 GB/s HDD read is slow but only affects first load. Subsequent runs read from page cache.
3. **cuDNN Conv3D workaround** — Was active (cuDNN 9.13.0 + torch 2.9.1 triggers `NVIDIA_MEMORY_CONV_BUG_WORKAROUND`). Upgraded to cuDNN 9.15.1 to clear it. Did NOT help — the workaround only affects VAE Conv3d decode, which wasn't the bottleneck.
4. **`--disable-mmap`** — Added 15s upfront (eager read into RAM) but didn't help warm runs.
5. **CPU→GPU transfer bandwidth** — Measured at 2.26 GB/s. 8.8GB would transfer in ~4s. Not the bottleneck.
6. **`--cache-classic`** — Changes node output caching strategy, not model loading. The speedup came from `--disable-dynamic-vram`.

---

## Issue 2: cuDNN Conv3D Workaround (Resolved)

### Symptom
After upgrading torch to 2.9.1+cu130 and driver to 580.173.02, `NVIDIA_MEMORY_CONV_BUG_WORKAROUND` was activated.

### Cause
cuDNN 9.13.0 (torch 2.9.1+cu130's bundled version) falls in the affected range:
```python
# comfy/ops.py:105
if (cudnn_version >= 91002 and cudnn_version < 91500) and \
   torch_version_numeric >= (2, 9) and torch_version_numeric <= (2, 10):
    NVIDIA_MEMORY_CONV_BUG_WORKAROUND = True
```

This replaces the normal Conv3d forward with a raw `torch.cudnn_convolution` call:
```python
# comfy/ops.py:614-618
if NVIDIA_MEMORY_CONV_BUG_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
    out = torch.cudnn_convolution(input, weight, self.padding, self.stride,
                                  self.dilation, self.groups, benchmark=False,
                                  deterministic=False, allow_tf32=True)
```

### Fix
Upgraded cuDNN from 9.13.0.50 to 9.15.1.9:
```bash
pip install nvidia-cudnn-cu13==9.15.1.9
```

cuDNN 9.15.1 (reports as 91501) is above the `< 91500` threshold, so the workaround no longer triggers. The torch dependency conflict warning is expected but safe due to cuDNN ABI compatibility.

### Impact
Minimal for Klein (affects only VAE Conv3d decode path). Would be more significant for models with heavier Conv3d usage.

---

## Issue 3: MiniMax H3 — 20GB on 12GB VRAM

### Status
Resolved (2026-09-08). Working config found. MiniMax H3 is 20GB (int8 quantized) on a 12GB GPU. DynamicVRAM is **mandatory**; the legacy patcher path cannot be used with H3 at all (it freezes the machine — see Issue 4).

### Recommendation (working startup flags)

DynamicVRAM is the correct tool for models that don't fit in VRAM. The per-prompt overhead is a ComfyUI design limitation that would need a code fix (caching VBAR setup state across identical prompts) to resolve properly.

**Working startup line for MiniMax H3 (RTX 3060 12GB / 32GB RAM):**
```
./venv/bin/python main.py --listen --vram-headroom 2.0 --cache-ram 7 7
```

- **Do NOT use `--disable-dynamic-vram` with H3.** The legacy LOWVRAM path materializes the LoRA-merged DiT (~20GB) and the CLIP encoder (~15GB) as *non-evictable* RAM tensors simultaneously — ~35GB on a 32GB machine → RAM collapse + swap thrash + machine freeze (Issue 4). DynamicVRAM (default on, via `comfy-aimdo`) evicts RAM properly.
- `--vram-headroom 2.0` keeps 2GB of VRAM free for the activation spike during DiT layer forwards (the anti-OOM knob; legacy equivalent is `--reserve-vram 2.0`). It does **not** force the model to RAM — the 20GB model streams from RAM regardless, but the headroom stops `torch.cuda.OutOfMemoryError` mid-step.
- `--cache-ram 7 7` — both values are *free-RAM floors*, not cache sizes (see Issue 4 for proof and details). `7 7` = keep ≥7GB of RAM free for the OS at all times.

### Measured performance (2026-09-08, RTX 3060 12GB / 32GB RAM, 7200rpm HDD)

| Run | Config | Wall time | Result |
|-----|--------|-----------|--------|
| Staged 3×10s → ~30s master (243f/shot, 640×640, seed 42) | ref2v turbo 4st v0.1 @ 0.75, 4 steps | **26m06s** (1566 s) | Completed end-to-end, RAM stable |
| Single-shot 3.4s test (81f, 640×640) | fl2v turbo 8st v1.0 @ 1.0, 8 steps | 10m11s (611 s) | Completed; ~100s was model load + CLIP encode, the rest sampling |

During sampling at dynamic VRAM: GPU util ~100%, VRAM in-use ~6-7GB (weight streaming), RAM available bottomed at ~3.2GB (floor held), swap used held at ~1.4GB (no thrash). First load is slow because the 20GB weights come off a 0.23GB/s HDD; subsequent runs benefit from page cache.

### Quality findings (which turbo LoRA to use)

- `minimax_h3_ref2v_turbo_4step_v0.1` is a **Ref2VA-only** distill — it needs reference conditioning (character/pose/motion/voice) and the Ref2VA graph. Used on a plain text-to-video FL2VA workflow it produced visibly poor output (soft/blurry, "not shareable").
- The correct turbo checkpoint for plain T2V on the FL2VA base is **`minimax_h3_fl2v_turbo_8step_v1.0` at strength 1.0, 8 steps, `res_multistep`/`simple`** — verified much better (user: "really good").
- Always validate quality on a short single-shot clip before committing to a full 3×10s run (~26-40min).

### Silver H3 Batch node (comfyui-silver-nodes)

Given that each MiniMax H3 generation pays heavy model-swap costs (the model doesn't fit in VRAM, so the system pages modules in/out around sampling), the `SilverH3Batch` node exists to **amortize those swaps across a batch of clips** rather than paying them per-clip:

- Queuing the minimax node N times pays `~3 x N` model swaps: text encoder -> DiT -> VAE per clip (each clip re-loads the TE to encode, the DiT to sample, the VAE to decode).
- `SilverH3Batch` stages the work into three phases so each model loads exactly **once for the whole batch**:
  - **Phase A — ENCODE**: every prompt text-encoded in one TE session; start frames / character image / character voice / guide audio encoded once and reused.
  - **Phase B — SAMPLE**: DiT hot, every video sampled back-to-back (no chaining / handoff unless combining).
  - **Phase C — DECODE + SAVE**: VAEs hot, each latent decoded, streamed to its own mp4 and freed immediately — at most ONE clip's frames are resident.
- Net: **2 model swaps per batch** (TE -> DiT, DiT -> VAE) instead of `~3 x N`, and RAM stays ~one clip regardless of batch size. The `_evict()` helper performs a controlled VRAM sweep at each phase boundary, never mid-phase.
- This directly mitigates the 20GB-on-12GB problem: it doesn't make the model fit, it removes the repeated load/unload churn that a naive per-clip queue would cause on a tightly-paged GPU.
- Implementation: `custom_nodes/comfyui-silver-nodes/silver_h3_batch.py` (and `silver_h3_staged.py` for the chained multishot variant).
- Without dynamic VRAM, the classic fallback would still thrash model weights during each video — the node's value is in keeping the loaded models resident across all N clips.

---

## Issue 4: MiniMax H3 — RAM collapse / system freeze on `--disable-dynamic-vram` (resolved)

### Symptom
With the legacy patcher (`--disable-dynamic-vram`), H3 runs drove RAM to 0 and the whole system became unresponsive (kernel swap thrash) mid-generation.

### Run history (attempt log)

| Prompt | Flags | Outcome |
|--------|-------|---------|
| `17c3959e` | dynamic ON (`--reserve-vram 2.0` online unused) + `--cache-ram 25.0` | Slow (>1min/step), canceled — no freeze |
| `32c6ab68` | `--disable-dynamic-vram --reserve-vram 2.0 --cache-ram 25.0` | Freeze: RAM 30.6/32GB used, 232MB free, swap 11GB used (pswpin 134350 / pswpout 2954515) |
| `6a40c75b` | `--disable-dynamic-vram --reserve-vram 2.0 --cache-ram 7 7` | Freeze again: ramAvail 19→15→5.7→4.2→1.4→0.2GB, swap 1.26→2.9GB, API dead at t≈438s |
| `ce4fa882` | dynamic ON + `--vram-headroom 2.0 --cache-ram 7 7` | ✅ Complete 26m06s, RAM stable |

### Root cause
Two independent things, both on the legacy path:

1. **Model weights are non-evictable RAM.** `--disable-dynamic-vram` uses the legacy `ModelPatcher` + LOWVRAM. It holds the *LoRA-merged* DiT (~20GB) and the CLIP encoder (~15GB) in RAM as real tensors at the same time → ~35GB demand on 32GB → swap → freeze.
2. **`--cache-ram` floors fix none of it.** Both `--cache-ram` values are *free-RAM floors* for **node cache entries only** (`execution.py:746,800-812`, `comfy_execution/caching.py:550-566`). They never touch model weights. A single value `25.0` sets the *second* floor (inactive/pinned cache) to effectively unbounded (`main.py:331-349`), so the pinned cache grows until RAM is exhausted — but even `--cache-ram 7 7` cannot prevent the weights-driven collapse. This is why both legacy runs froze.

### Fix
Use DynamicVRAM (default). `cache-ram 7 7` keeps ≥7GB free for the OS; DynamicVRAM's RAM eviction keeps the workspace livable. `--disable-dynamic-vram` is now only used for models that fully fit VRAM (see Klein 9B, Issue 1).

### Guardrail
When running long H3 gens, watch a RAM safety tripwire: if `MemAvailable` drops below ~2GB, POST `/interrupt`; if it stays there >25s, SIGTERM the `main.py` process so the machine never hard-freezes. (`/tmp/opencode/memwatch.py` samples RAM/VRAM/swap; `/tmp/opencode/watchdog_watcher.py` auto-interrupts.)

---

## Key Code Locations

| Concern | File | Lines |
|---------|------|-------|
| DynamicVRAM per-prompt load() | `comfy/model_patcher.py` | 2141-2158 (partially_load), 1853-2035 (load) |
| Per-module VBAR setup loop | `comfy/model_patcher.py` | 1911-2010 |
| `_load_list()` module iteration | `comfy/model_patcher.py` | 945-980 |
| `dirty` flag check | `comfy/model_patcher.py` | 2144 |
| `_v_signature` preservation | `comfy/model_patcher.py` | 1914-1916 (set_dirty) |
| VBAR fault + pin + DMA | `comfy/ops.py` | 126-264 (cast_modules_with_vbar) |
| Conv3D workaround | `comfy/ops.py` | 101-110, 607-620 |
| DynamicVRAM init | `main.py` | 251-282 |
| Legacy vs Dynamic patcher swap | `comfy/model_patcher.py` | 2178 |
| Pin eviction between prompts | `comfy/model_management.py` | 647-736 |
| MAX_PINNED_MEMORY calc | `comfy/model_management.py` | 1581-1588 |
| load_models_gpu orchestration | `comfy/model_management.py` | 909-1009 |
| free_memory / model eviction | `comfy/model_management.py` | 863-907 |
