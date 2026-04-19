"""
Modal GPU functions — all hardware work runs here on an A100-80GB.

Functions exposed to the agent:
  profile_kernel(kernel_code, config)  →  hardware metrics dict
  get_kernel_metadata(kernel_code)     →  compiled IR stats

Functions for the inference demo:
  run_inference_comparison(prompt)     →  async generator streaming tokens
"""
import importlib.util
import math
import os
import sys
import tempfile
import traceback

import modal


def _write_and_import_kernel(kernel_code: str):
    """
    Write kernel code to a temp file and import it as a module.

    Required because @triton.jit uses inspect.getsource() internally to read
    the function source for PTX compilation. exec()-defined functions have no
    source file, so getsource() raises 'could not get source code'.
    Writing to disk first solves this completely.

    Returns (module, tmp_path). Caller is responsible for os.unlink(tmp_path).
    """
    # Strip UTF-8 BOM if present (PowerShell Set-Content adds one)
    kernel_code = kernel_code.lstrip('\ufeff')

    # Prepend standard imports so agent-generated code can omit them
    preamble = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "import math\n\n"
    )
    full_source = preamble + kernel_code

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp", prefix="attn_kernel_"
    )
    tmp.write(full_source)
    tmp.close()

    spec = importlib.util.spec_from_file_location("_dyn_attn_kernel", tmp.name)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["_dyn_attn_kernel"] = mod
    spec.loader.exec_module(mod)
    return mod, tmp.name

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App("qwen3-kernel-optimizer")

# Persist downloaded model weights across runs — first run downloads, all others load locally
model_cache = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/model-cache"

# Use the official PyTorch CUDA image as the base so torch.cuda is always available.
# debian_slim + pip_install("torch") can pick up the CPU-only wheel on some builds.
gpu_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("gcc", "build-essential")   # Triton JIT needs gcc to compile cuda_utils.so
    .pip_install(
        "triton==2.3.0",
        "transformers>=4.51.0",   # Qwen3 requires 4.51+
        "accelerate>=0.30.0",
        "numpy<2.0",
        "huggingface_hub",
    )
)

MODEL_ID = "Qwen/Qwen3-4B"

# A100 hardware constants used for roofline analysis
A100_PEAK_TFLOPS_FP16 = 312.0   # Tensor Core FP16
A100_PEAK_BANDWIDTH   = 2000.0  # GB/s HBM


# ---------------------------------------------------------------------------
# Kernel profiling tool
# ---------------------------------------------------------------------------

@app.function(gpu="A100-80GB", image=gpu_image, timeout=180)
def profile_kernel(kernel_code: str, config: dict) -> dict:
    """
    Execute and profile a Triton attention kernel on A100.

    kernel_code must define:
        def attention_kernel(q, k, v, is_causal=False, scale=None) -> torch.Tensor

    config keys:
        seq_len  (int)  — sequence length, default 1024
        d_head   (int)  — head dimension, default 128
        n_heads  (int)  — number of query heads, default 32
        batch    (int)  — batch size, default 2

    Returns a metrics dict (see below).
    """
    import torch
    import triton
    import triton.language as tl

    seq_len = config.get("seq_len", 1024)
    d_head  = config.get("d_head",  128)
    n_heads = config.get("n_heads", 32)
    batch   = config.get("batch",   2)

    print(f"[profile_kernel] seq={seq_len} d={d_head} heads={n_heads} batch={batch}")

    # --- Write kernel to temp file and import (exec() breaks triton.jit's getsource) ---
    tmp_path = None
    try:
        mod, tmp_path = _write_and_import_kernel(kernel_code)
        attention_kernel = getattr(mod, "attention_kernel", None)
        if attention_kernel is None:
            return {"success": False, "error": "attention_kernel not defined in kernel code", "phase": "exec"}
        print("[profile_kernel] Kernel imported OK")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[profile_kernel] Import failed:\n{tb}")
        return {"success": False, "error": f"Import failed: {e}\n{tb}", "phase": "exec"}

    # --- Build test tensors ---
    device = "cuda"
    dtype  = torch.float16
    q = torch.randn(batch, n_heads, seq_len, d_head, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # --- Correctness check vs PyTorch reference (both causal and non-causal) ---
    def _pt_reference(q_, k_, v_, is_causal_):
        scale_ = 1.0 / math.sqrt(d_head)
        scores_ = torch.matmul(q_.float(), k_.float().transpose(-2, -1)) * scale_
        if is_causal_:
            N_ = q_.shape[-2]
            mask_ = torch.tril(torch.ones(N_, N_, device=q_.device, dtype=torch.bool))
            scores_ = scores_.masked_fill(~mask_, float("-inf"))
        return torch.matmul(torch.softmax(scores_, dim=-1), v_.float()).to(dtype)

    max_err = 0.0
    try:
        for is_causal in (False, True):
            label = "causal" if is_causal else "non-causal"
            print(f"[profile_kernel] Correctness check ({label})...")
            ref_out    = _pt_reference(q, k, v, is_causal)
            kernel_out = attention_kernel(q, k, v, is_causal=is_causal)
            torch.cuda.synchronize()
            err = (kernel_out.float() - ref_out.float()).abs().max().item()
            max_err = max(max_err, err)
            print(f"[profile_kernel] {label} max_err={err:.6f}")
            if err >= 0.05:
                return {
                    "success":   False,
                    "error":     f"Output mismatch ({label}): max_error={err:.4f} (threshold 0.05)",
                    "phase":     "correctness",
                    "max_error": err,
                }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[profile_kernel] Correctness check failed:\n{tb}")
        return {"success": False, "error": f"Correctness check failed: {e}\n{tb}", "phase": "correctness"}

    correct = True
    print(f"[profile_kernel] Both causal+non-causal correct. max_err={max_err:.6f}")

    # --- Performance benchmark ---
    try:
        print("[profile_kernel] Benchmarking...")
        ms = triton.testing.do_bench(lambda: attention_kernel(q, k, v), warmup=20, rep=30)
        print(f"[profile_kernel] time={ms:.4f}ms")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[profile_kernel] Benchmark failed:\n{tb}")
        return {"success": False, "error": f"Benchmark failed: {e}\n{tb}", "phase": "benchmark"}

    # --- Triton compile metadata (free; super useful for diagnosing occupancy) ---
    kernel_meta = _collect_kernel_metadata(mod)
    print(f"[profile_kernel] kernel_meta={kernel_meta}")

    # --- torch.profiler: per-kernel device time + launch info ---
    torch_prof_data = _torch_profile_kernel(attention_kernel, q, k, v)
    print(f"[profile_kernel] torch_prof={torch_prof_data}")

    # --- Triton Proton: HBM bytes, tensor core utilization (graceful fallback) ---
    proton_data = _proton_profile_kernel(attention_kernel, q, k, v)
    print(f"[profile_kernel] proton={proton_data}")

    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    # --- Roofline math ---
    total_flops = 4 * batch * n_heads * (seq_len ** 2) * d_head
    total_bytes = 4 * batch * n_heads * seq_len * d_head * 2

    tflops             = total_flops / (ms * 1e-3) / 1e12
    bandwidth_gbs      = total_bytes / (ms * 1e-3) / 1e9
    efficiency_pct     = tflops / A100_PEAK_TFLOPS_FP16 * 100
    bandwidth_util_pct = bandwidth_gbs / A100_PEAK_BANDWIDTH * 100
    arith_intensity    = total_flops / total_bytes
    ridge_point        = A100_PEAK_TFLOPS_FP16 * 1e12 / (A100_PEAK_BANDWIDTH * 1e9)
    bound = "memory" if arith_intensity < ridge_point else "compute"

    # --- Analytical occupancy estimate from compile metadata ---
    occupancy = _estimate_occupancy(kernel_meta)

    return {
        "success":               True,
        "time_ms":               round(ms, 4),
        "tflops":                round(tflops, 3),
        "bandwidth_gbs":         round(bandwidth_gbs, 1),
        "efficiency_pct":        round(efficiency_pct, 2),
        "bandwidth_util_pct":    round(bandwidth_util_pct, 2),
        "arithmetic_intensity":  round(arith_intensity, 1),
        "ridge_point_flop_byte": round(ridge_point, 1),
        "bound":                 bound,
        "max_error":             round(max_err, 6),
        "config":                config,
        # Compile metadata (per-kernel)
        "kernel_metadata":       kernel_meta,
        "occupancy":             occupancy,
        # Real hardware profiling
        "torch_profile":         torch_prof_data,
        "proton_profile":        proton_data,
        # Hardware ceilings for agent reference
        "a100_peak_tflops":      A100_PEAK_TFLOPS_FP16,
        "a100_peak_bandwidth":   A100_PEAK_BANDWIDTH,
    }


# ---------------------------------------------------------------------------
# Profiling helpers (kernel metadata, torch.profiler, Proton)
# ---------------------------------------------------------------------------

def _collect_kernel_metadata(mod) -> list:
    """
    Scan an imported kernel module for Triton JIT functions and extract their
    compile metadata: registers/thread, spills, shared memory.
    Returns a list of dicts, one per compiled kernel variant.
    """
    import triton
    results = []
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if not isinstance(attr, triton.runtime.jit.JITFunction):
            continue
        # Triton caches compiled variants keyed by device -> signature
        for device_cache in getattr(attr, "cache", {}).values():
            for compiled in device_cache.values():
                meta = getattr(compiled, "metadata", None)
                # Handle both old (dict) and new (namespace) metadata
                def pick(key, default=None):
                    if meta is None:
                        return getattr(compiled, key, default)
                    if isinstance(meta, dict):
                        return meta.get(key, getattr(compiled, key, default))
                    return getattr(meta, key, getattr(compiled, key, default))
                entry = {
                    "kernel":    attr_name,
                    "n_regs":    pick("n_regs"),
                    "n_spills":  pick("n_spills"),
                    "shared":    pick("shared"),        # bytes
                    "num_warps": pick("num_warps"),
                    "num_stages": pick("num_stages"),
                }
                if any(v is not None for v in entry.values() if not isinstance(v, str)):
                    results.append(entry)
    return results


def _estimate_occupancy(kernel_meta: list) -> dict:
    """
    Compute analytical A100 occupancy from compile metadata.
    A100 per-SM limits: 65536 registers, 192 KB shared mem, 2048 threads, 32 blocks.
    """
    if not kernel_meta:
        return {"blocks_per_sm": None, "warps_per_sm": None, "limiter": "unknown"}

    # Use the kernel with the highest reg count (the hot one)
    hot = max(kernel_meta, key=lambda k: (k.get("n_regs") or 0))
    regs      = hot.get("n_regs") or 0
    shared    = hot.get("shared") or 0
    num_warps = hot.get("num_warps") or 4
    threads_per_block = num_warps * 32

    blocks_by_regs   = 65536 // max(regs * threads_per_block, 1) if regs else 32
    blocks_by_shared = (192 * 1024) // max(shared, 1) if shared else 32
    blocks_by_threads = 2048 // max(threads_per_block, 1)
    blocks_per_sm = min(blocks_by_regs, blocks_by_shared, blocks_by_threads, 32)

    limiter = "registers"
    if blocks_by_shared < blocks_by_regs and blocks_by_shared < blocks_by_threads:
        limiter = "shared_memory"
    elif blocks_by_threads < blocks_by_regs and blocks_by_threads < blocks_by_shared:
        limiter = "threads"

    return {
        "blocks_per_sm":   int(blocks_per_sm),
        "warps_per_sm":    int(blocks_per_sm * num_warps),
        "max_warps_per_sm": 64,
        "occupancy_pct":   round(blocks_per_sm * num_warps / 64 * 100, 1),
        "limiter":         limiter,
    }


def _torch_profile_kernel(fn, q, k, v) -> dict:
    """
    Run the kernel under torch.profiler to get precise per-kernel CUDA device time.
    Returns {device_time_ms, kernel_names} or {"error": "..."} on failure.
    """
    import torch
    try:
        from torch.profiler import profile, ProfilerActivity
        # Warmup outside the profiler to keep cache compilation out of the trace
        for _ in range(3):
            fn(q, k, v)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(10):
                fn(q, k, v)
            torch.cuda.synchronize()

        events = prof.key_averages()
        # Sum device time across all Triton kernel events
        total_cuda_us = 0
        kernel_names = []
        for e in events:
            dev_us = getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0)
            if dev_us and ("triton" in e.key.lower() or "fwd" in e.key.lower() or "attn" in e.key.lower()):
                total_cuda_us += dev_us
                kernel_names.append({"name": e.key[:60], "count": e.count, "dev_ms": round(dev_us / 1000, 4)})
        return {
            "device_time_ms_per_call": round(total_cuda_us / 1000 / 10, 4),
            "kernels":                 kernel_names[:5],  # cap output size
        }
    except Exception as e:
        return {"error": f"torch.profiler failed: {type(e).__name__}: {e}"}


def _proton_profile_kernel(fn, q, k, v) -> dict:
    """
    Run the kernel under Triton Proton to get HBM bytes and tensor core FLOPs.
    Graceful fallback if Proton is unavailable or API differs.
    """
    try:
        import triton.profiler as proton
    except Exception as e:
        return {"error": f"Proton unavailable: {e}"}

    import tempfile
    try:
        import torch
        # Warmup
        for _ in range(3):
            fn(q, k, v)
        torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(suffix=".hatchet", delete=False) as tmp_trace:
            trace_path = tmp_trace.name

        # Start Proton with hatchet backend (simplest to parse)
        proton.start(trace_path, backend="hatchet", context="shadow")
        for _ in range(5):
            with proton.scope("attention"):
                fn(q, k, v)
        torch.cuda.synchronize()
        proton.finalize()

        # Parse the trace file (JSON-like hatchet format)
        import json
        with open(trace_path) as f:
            trace = json.load(f)

        # Extract per-scope metrics (flops, bytes, time)
        summary = {"scopes": []}
        def walk(node, parent=""):
            name = node.get("frame", {}).get("name", "")
            metrics = node.get("metrics", {})
            if metrics:
                summary["scopes"].append({
                    "name":   name,
                    "time_ns": metrics.get("time (ns)", 0),
                    "flops":   metrics.get("flops", 0),
                    "bytes":   metrics.get("bytes", 0),
                })
            for child in node.get("children", []):
                walk(child, name)
        walk(trace if isinstance(trace, dict) else trace[0])
        os.unlink(trace_path)
        return summary
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": f"Proton profile failed: {type(e).__name__}: {e}", "traceback": tb[-400:]}


# ---------------------------------------------------------------------------
# Kernel metadata tool (compiled IR stats)
# ---------------------------------------------------------------------------

@app.function(gpu="A100-80GB", image=gpu_image, timeout=120)
def get_kernel_metadata(kernel_code: str, d_head: int = 128) -> dict:
    """
    Compile the kernel and return Triton IR metadata:
    shared memory usage, register count, warp count.
    Useful for the agent to understand resource utilization.
    """
    import torch

    tmp_path = None
    try:
        mod, tmp_path = _write_and_import_kernel(kernel_code)
        fn = getattr(mod, "attention_kernel", None)
        if fn is None:
            return {"success": False, "error": "attention_kernel not defined"}

        q = torch.randn(1, 1, 64, d_head, device="cuda", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        fn(q, k, v)
        torch.cuda.synchronize()
        print(f"[get_kernel_metadata] Compiled OK d_head={d_head}")
        return {"success": True, "message": "Kernel compiled successfully", "d_head": d_head}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[get_kernel_metadata] Failed:\n{tb}")
        return {"success": False, "error": f"{e}\n{tb}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Side-by-side inference comparison (streaming generator)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    timeout=600,
    volumes={MODEL_CACHE_DIR: model_cache},
)
def run_inference_comparison(prompt: str, kernel_code: str, max_new_tokens: int = 150):
    """
    Load Qwen3-4B, run baseline SDPA inference, then run Triton-patched inference.

    kernel_code is the agent's best-found kernel (exec'd directly in this container).
    Passing it as a parameter avoids any filesystem mounting.

    Yields dicts:
        {"phase": "loading"}
        {"phase": "baseline_done", "text": str, "tokens_per_sec": float, "time_ms": float}
        {"phase": "triton_token",  "token": str, "tokens_per_sec": float, "elapsed_ms": float}
        {"phase": "triton_done",   "tokens_per_sec": float, "time_ms": float, "speedup": float}
        {"phase": "error",         "message": str}
    """
    import time
    from threading import Thread

    import torch
    import torch.nn.functional as F
    import triton
    import triton.language as tl
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    yield {"phase": "loading"}

    try:
        import os
        # Load from volume cache if available, else download and cache
        cache_dir = MODEL_CACHE_DIR
        local_only = os.path.isdir(os.path.join(cache_dir, "models--" + MODEL_ID.replace("/", "--")))
        load_kwargs = dict(
            cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=local_only,   # skip hash check / network on warm runs
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="sdpa",
            **load_kwargs,
        )
        model.eval()
        if not local_only:
            model_cache.commit()   # persist newly downloaded weights to the volume
    except Exception as e:
        yield {"phase": "error", "message": f"Model load failed: {e}"}
        return

    # Apply Qwen3 chat template — required for correct output (it's a chat model,
    # not a completion model; raw prompt input causes repetition/garbage)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
    }

    # ── Phase 1: Baseline (standard PyTorch SDPA) ──────────────────────────
    yield {"phase": "baseline_start"}
    try:
        # Use streamer for baseline too so we can capture TTFT
        baseline_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        baseline_thread   = Thread(
            target=model.generate,
            kwargs={**gen_kwargs, "streamer": baseline_streamer},
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        baseline_thread.start()

        baseline_ttft_ms = None
        baseline_text    = ""
        baseline_tokens  = 0
        for tok in baseline_streamer:
            if baseline_ttft_ms is None:
                baseline_ttft_ms = (time.perf_counter() - t0) * 1000
            baseline_text   += tok
            baseline_tokens += 1

        baseline_thread.join()
        torch.cuda.synchronize()
        baseline_ms  = (time.perf_counter() - t0) * 1000
        baseline_tps = baseline_tokens / (baseline_ms / 1000)

        print(f"[baseline] ttft={baseline_ttft_ms:.1f}ms total={baseline_ms:.1f}ms tps={baseline_tps:.1f}")

        yield {
            "phase":          "baseline_done",
            "text":           baseline_text,
            "tokens_per_sec": round(baseline_tps, 1),
            "time_ms":        round(baseline_ms, 1),
            "ttft_ms":        round(baseline_ttft_ms or 0, 1),
            "token_count":    baseline_tokens,
        }
    except Exception as e:
        yield {"phase": "error", "message": f"Baseline failed: {e}"}
        return

    # ── Phase 2: Triton-patched (streaming) ────────────────────────────────
    yield {"phase": "triton_start"}
    try:
        import queue as _queue

        _original_sdpa = F.scaled_dot_product_attention
        patched = False

        if kernel_code:
            try:
                _mod, _tmp = _write_and_import_kernel(kernel_code)
                _triton_attn = getattr(_mod, "attention_kernel", None)

                if _triton_attn is not None:
                    # Warmup: compile BOTH the non-causal and causal kernel variants
                    # before the generation thread starts.
                    #
                    # Triton JIT compiles a separate binary for each unique combination
                    # of constexpr arguments (STAGE, BLOCK_M, BLOCK_N, HEAD_DIM, ...).
                    # is_causal=True maps to a different STAGE value, so it needs its
                    # own compiled binary.  Qwen3 uses causal attention for every layer,
                    # so if we only warm up is_causal=False the causal binary gets
                    # compiled inside model.generate() on the background thread.  If
                    # that compilation or the resulting kernel fails, the thread dies
                    # silently and TextIteratorStreamer waits until the 60-s timeout.
                    #
                    # Warming up both variants here ensures any compilation failure is
                    # caught as a Python exception before the generation thread starts.
                    try:
                        _wq = torch.randn(1, 1, 128, 128, device="cuda", dtype=torch.float16)
                        _triton_attn(_wq, _wq, _wq, is_causal=False)
                        _triton_attn(_wq, _wq, _wq, is_causal=True)
                        torch.cuda.synchronize()
                        print("[triton] Kernel warmup OK (causal + non-causal)")
                        del _wq
                    except Exception as warmup_err:
                        print(f"[warn] Kernel warmup failed — will not patch SDPA: {warmup_err}")
                        _triton_attn = None

                if _triton_attn is not None:
                    # _hook_failed: disables kernel after first error to prevent
                    # repeated CUDA error cascades corrupting the context.
                    # _hook_calls: used to synchronize exactly once after the very
                    # first real Triton call, turning any delayed CUDA error into a
                    # catchable Python exception before it can corrupt subsequent ops.
                    _hook_failed = [False]
                    _hook_calls  = [0]

                    def _sdpa_hook(query, key, value, attn_mask=None,
                                   dropout_p=0.0, is_causal=False, scale=None):
                        B, H_q, N, D = query.shape
                        H_kv = key.shape[1]
                        # Only intercept prefill (N >= 64): decode steps (N=1) fall
                        # back to PyTorch SDPA which is more efficient for single-token
                        # queries against a KV cache. Also restrict to fp16, supported
                        # head dims, no attn_mask/dropout. Disabled after first error.
                        if (_hook_failed[0]
                                or query.dtype != torch.float16
                                or D not in (64, 128)
                                or dropout_p > 0
                                or attn_mask is not None
                                or N < 64
                                or N > 8192):
                            return _original_sdpa(query, key, value, attn_mask,
                                                  dropout_p, is_causal, scale)
                        # GQA expansion: repeat KV heads to match Q heads
                        if H_kv != H_q:
                            n_rep = H_q // H_kv
                            key   = key.repeat_interleave(n_rep, dim=1)
                            value = value.repeat_interleave(n_rep, dim=1)
                        try:
                            result = _triton_attn(query, key, value,
                                                  is_causal=is_causal, scale=scale)
                            # Synchronize once after the very first Triton call so
                            # any delayed CUDA error (illegal memory access, etc.)
                            # surfaces as a Python exception here rather than
                            # corrupting the CUDA context for subsequent operations.
                            if _hook_calls[0] == 0:
                                torch.cuda.synchronize()
                            _hook_calls[0] += 1
                            return result
                        except Exception as hook_err:
                            print(f"[warn] Triton hook error — disabling for this run: {hook_err}")
                            _hook_failed[0] = True
                            return _original_sdpa(query, key, value, attn_mask,
                                                  dropout_p, is_causal, scale)

                    F.scaled_dot_product_attention = _sdpa_hook
                    patched = True
            except Exception as e:
                print(f"[warn] Kernel import failed, falling back to SDPA: {e}")

        # timeout=60: if the generation thread dies without signalling the streamer
        # (e.g. CUDA context corrupted by a kernel error), the for-loop raises
        # queue.Empty instead of blocking forever.
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
        )
        # daemon=True: if the thread hangs (Triton deadlock), it won't prevent
        # the Modal container from exiting and returning an error to the client.
        thread = Thread(
            target=model.generate,
            kwargs={**gen_kwargs, "streamer": streamer},
            daemon=True,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        thread.start()

        token_count    = 0
        triton_ttft_ms = None
        timed_out      = False
        try:
            for token_text in streamer:
                if triton_ttft_ms is None:
                    triton_ttft_ms = (time.perf_counter() - t0) * 1000
                token_count += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000
                tps = token_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
                yield {
                    "phase":          "triton_token",
                    "token":          token_text,
                    "tokens_per_sec": round(tps, 1),
                    "elapsed_ms":     round(elapsed_ms, 1),
                }
        except _queue.Empty:
            timed_out = True
            print("[triton] Streamer timed out — generation thread likely hung or crashed.")

        if patched:
            F.scaled_dot_product_attention = _original_sdpa

        if timed_out:
            yield {
                "phase":   "error",
                "message": (
                    "Triton generation timed out after 60 s. The kernel may be hanging "
                    "on this model's attention shapes. Partial output discarded."
                ),
            }
            return

        thread.join(timeout=5)
        torch.cuda.synchronize()
        total_ms  = (time.perf_counter() - t0) * 1000
        final_tps = token_count / (total_ms / 1000) if total_ms > 0 else 0

        print(f"[triton] ttft={triton_ttft_ms:.1f}ms total={total_ms:.1f}ms tps={final_tps:.1f}")

        yield {
            "phase":          "triton_done",
            "tokens_per_sec": round(final_tps, 1),
            "time_ms":        round(total_ms, 1),
            "ttft_ms":        round(triton_ttft_ms or 0, 1),
            "token_count":    token_count,
            "speedup":        round(final_tps / baseline_tps, 2) if baseline_tps > 0 else 1.0,
            "ttft_speedup":   round((baseline_ttft_ms or 1) / (triton_ttft_ms or 1), 2),
        }

    except Exception as e:
        if patched:
            F.scaled_dot_product_attention = _original_sdpa
        yield {"phase": "error", "message": f"Triton inference failed: {e}"}
