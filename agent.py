"""
Kernel optimization agent — Llama 3.3 70B via Groq (configurable).

Uses a generate-then-profile loop: the LLM outputs an improved kernel in a
```python block, we extract it, profile it on Modal A100, and feed the results back.
No tool-calling required -- avoids provider-specific compatibility issues.

This "generate → profile → feedback" loop is simpler and 100% reliable.

Motus (https://www.lithosai.com/) wraps this agent as its serving harness.
Every optimization trace is logged so Motus can learn which strategies work
for which bottleneck types and surface that context on future runs.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Generator

import modal
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from prompts.optimizer import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# LLM client — OpenAI-compatible, provider selected by LLM_PROVIDER below.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model config — swap LLM_PROVIDER to change backends, all are OpenAI-compatible.
#
#   groq    (DEFAULT) Llama-3.3-70B  — 14,400 req/day free, fast, great at code
#   gemini             gemini-2.0-flash — 1500 req/day free
# ---------------------------------------------------------------------------

LLM_PROVIDER = "claude"   # "gemini" | "claude" | "deepseek" | "cerebras" | "groq" | "sambanova"

_PROVIDER_CONFIGS = {
    "claude": {
        "env_key":  "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
        "model":    "claude-sonnet-4-6",  # best coding model; ~$0.10/run
        "sign_up":  "https://console.anthropic.com",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "cerebras": {
        "env_key":  "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "model":    "qwen-3-235b-a22b-instruct-2507",
        "sign_up":  "https://console.cerebras.ai",
    },
    "groq": {
        "env_key":  "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama-3.3-70b-versatile",
        "sign_up":  "https://console.groq.com",
        # Free tier: 6,000 TPM, 30 RPM
    },
    "sambanova": {
        "env_key":  "SAMBANOVA_API_KEY",
        "base_url": "https://api.sambanova.ai/v1",
        "model":    "Meta-Llama-3.3-70B-Instruct",
        "sign_up":  "https://cloud.sambanova.ai",
    },
    "deepseek": {
        "env_key":  "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model":    "deepseek-chat",
        "sign_up":  "https://platform.deepseek.com",
    },
    "gemini": {
        "env_key":  "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model":    "models/gemma-4-31b-it",   # unlimited TPM, 15 RPM, 1500 RPD free
        "sign_up":  "https://aistudio.google.com",
    },
}

_cfg = _PROVIDER_CONFIGS[LLM_PROVIDER]
LLM_MODEL = _cfg["model"]


def _make_client() -> OpenAI:
    api_key = os.environ.get(_cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"{_cfg['env_key']} not set. "
            f"Get a key at {_cfg['sign_up']} and add it to your .env file."
        )
    return OpenAI(
        api_key=api_key,
        base_url=_cfg["base_url"],
        default_headers=_cfg.get("extra_headers", {}),
    )


# ---------------------------------------------------------------------------
# Motus trace logging
# ---------------------------------------------------------------------------

TRACES_PATH = Path(__file__).parent / ".motus_traces.jsonl"


def _log_motus_trace(trace: dict) -> None:
    with TRACES_PATH.open("a") as f:
        f.write(json.dumps(trace) + "\n")


def _load_motus_context() -> str:
    if not TRACES_PATH.exists():
        return ""
    traces = []
    with TRACES_PATH.open() as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    successful = [t for t in traces if t.get("speedup", 0) >= 2.0]
    if not successful:
        return ""
    recent = successful[-3:]
    lines = ["## Motus context: successful strategies from past runs\n"]
    for t in recent:
        lines.append(
            f"- Bottleneck: {t.get('bound', '?')}-bound | "
            f"Strategy: {t.get('winning_strategy', '?')} | "
            f"Speedup: {t.get('speedup', 0):.1f}x | "
            f"Block sizes: BLOCK_M={t.get('block_m', '?')}, BLOCK_N={t.get('block_n', '?')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modal function handles
# ---------------------------------------------------------------------------

def _call_profile_kernel(kernel_code: str, config: dict) -> dict:
    fn = modal.Function.from_name("qwen3-kernel-optimizer", "profile_kernel")
    return fn.remote(kernel_code, config)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "seq_len": 1024,
    "d_head":  128,
    "n_heads": 32,
    "batch":   2,
}

EFFICIENCY_TARGET_PCT = 70.0
MAX_ITERATIONS        = 6


def _extract_code_block(text: str) -> str | None:
    """
    Extract the first python code block from agent text.
    Handles both properly closed blocks and truncated responses (no closing ```).
    """
    # Prefer a fully closed block
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: take everything after the opening fence (truncated response)
    match = re.search(r"```python\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_param(code: str, name: str) -> str | None:
    match = re.search(rf"{name}\s*=\s*(\d+)", code)
    return match.group(1) if match else None


def _fmt_metrics(m: dict) -> str:
    if not m.get("success"):
        return f"FAILED: {m.get('error', 'unknown')}"

    base = (
        f"time_ms={m.get('time_ms', 0):.3f}, "
        f"tflops={m.get('tflops', 0):.2f} ({m.get('efficiency_pct', 0):.1f}% of 312), "
        f"bandwidth_gbs={m.get('bandwidth_gbs', 0):.0f} ({m.get('bandwidth_util_pct', 0):.1f}% of 1935), "
        f"bound={m.get('bound', '?')}, "
        f"AI={m.get('arithmetic_intensity', 0):.1f}"
    )

    # Compile metadata
    km = m.get("kernel_metadata") or []
    occ = m.get("occupancy") or {}
    if km:
        hot = max(km, key=lambda k: (k.get("n_regs") or 0))
        base += (
            f"\n  compile: regs={hot.get('n_regs')}/thread, "
            f"spills={hot.get('n_spills')}, "
            f"shared={hot.get('shared')}B, "
            f"warps={hot.get('num_warps')}, stages={hot.get('num_stages')}"
        )
    if occ.get("blocks_per_sm") is not None:
        base += (
            f"\n  occupancy: {occ.get('blocks_per_sm')} blocks/SM, "
            f"{occ.get('warps_per_sm')}/{occ.get('max_warps_per_sm')} warps "
            f"({occ.get('occupancy_pct')}%), limiter={occ.get('limiter')}"
        )

    # torch.profiler per-kernel device time
    tp = m.get("torch_profile") or {}
    if tp.get("device_time_ms_per_call") is not None:
        base += f"\n  torch.profiler: device_time={tp['device_time_ms_per_call']:.3f}ms/call"
    elif tp.get("error"):
        base += f"\n  torch.profiler: (n/a: {tp['error'][:80]})"

    # Proton
    pp = m.get("proton_profile") or {}
    scopes = pp.get("scopes") or []
    if scopes:
        attn = next((s for s in scopes if "attention" in s.get("name", "").lower()), scopes[0])
        time_ns = attn.get("time_ns") or 0
        flops   = attn.get("flops") or 0
        bytes_  = attn.get("bytes") or 0
        if time_ns > 0:
            meas_tflops = flops / time_ns / 1e3 if flops else 0
            meas_gbs    = bytes_ / time_ns if bytes_ else 0
            base += f"\n  proton: tflops={meas_tflops:.2f}, hbm_gbs={meas_gbs:.0f}, bytes={bytes_}"
    elif pp.get("error"):
        base += f"\n  proton: (n/a: {pp['error'][:80]})"

    return base


def _build_initial_message(kernel_code: str, baseline_metrics: dict, config: dict) -> str:
    motus_ctx   = _load_motus_context()
    motus_block = f"\n{motus_ctx}\n" if motus_ctx else ""

    return f"""\
Your task: write a Triton attention kernel that beats the PyTorch reference below.
{motus_block}
## PyTorch reference baseline (scores {baseline_metrics.get('tflops', '?'):.1f} TFLOPS -- your Triton kernel must exceed this)

```python
{kernel_code}
```

## Profiling results (seq_len={config['seq_len']}, d_head={config['d_head']}, n_heads={config['n_heads']}, batch={config['batch']})

{_fmt_metrics(baseline_metrics)}

{_diagnose(baseline_metrics)}

Write a Triton kernel from scratch that beats the PyTorch reference above.
Implement the FlashAttention-style tiled algorithm: process Q/K/V in BLOCK_M x BLOCK_N tiles,
maintain running max (m_i) and normaliser (l_i) for online softmax, accumulate output in SRAM.
This fuses the three HBM passes of the PyTorch reference into one, dramatically reducing
memory traffic and raising TFLOPS.

Rules:
- Output ONLY the complete Python source in a single ```python block.
- The code must define exactly one public function: attention_kernel(q, k, v, is_causal=False, scale=None)
- is_causal=True MUST be supported: skip upper-triangle K tiles and mask the last partial tile.
  For tiles where start_n + BLOCK_N <= start_m * BLOCK_M: skip entirely (causal means token i only attends to j<=i).
  For the last partial tile: apply mask qk = tl.where(offs_n[None,:] <= offs_m[:,None], qk, float("-inf"))
- Do NOT add import statements (torch, triton, tl, math are already in scope).
- Do NOT use tl.ones() -- it does not exist in Triton 2.3. Use tl.full([N], 1.0, dtype=tl.float32) instead.
- Do NOT use @triton.autotune. It causes too many crash patterns. Use fixed block sizes instead.
  Declare BLOCK_M and BLOCK_N as tl.constexpr in the @triton.jit kernel signature,
  then pass them in the grid launch along with num_warps and num_stages:
    RIGHT: _attn_fwd[grid](..., BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2)
- num_warps and num_stages are JIT meta-parameters -- NEVER declare them in the @triton.jit
  kernel signature. They are consumed by the Triton runtime, not passed into the kernel.
  If you put them in the signature you get "missing a required argument: num_warps".
  WRONG: def _attn_fwd(..., num_warps: tl.constexpr, num_stages: tl.constexpr):
  RIGHT: def _attn_fwd(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
- Use BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2 as your starting point.
- Use only ASCII characters in strings and comments (no em dashes, arrows, or Unicode).
- Keep the kernel concise -- aim for under 130 lines total. No verbose docstrings or long comments.

Target: {EFFICIENCY_TARGET_PCT}% compute efficiency ({EFFICIENCY_TARGET_PCT/100*312:.0f} TFLOPS).
"""


def _diagnose(result: dict) -> str:
    """
    Translate raw profiler numbers into a concrete, actionable diagnosis
    so the LLM knows exactly what lever to pull next.

    Layers:
      1. Roofline (compute vs memory bound).
      2. Compile metadata (regs, spills, shared mem) -> occupancy.
      3. torch.profiler device time discrepancy -> launch/sync overhead.
      4. Proton measured HBM bytes/FLOPs vs theoretical -> cache behavior.
    """
    lines = []
    bw     = result.get("bandwidth_util_pct", 0)
    eff    = result.get("efficiency_pct", 0)
    ai     = result.get("arithmetic_intensity", 0)
    ridge  = result.get("ridge_point_flop_byte", 156)
    bound  = result.get("bound", "memory")
    tflops = result.get("tflops", 0)

    # ---- Layer 1: roofline ----
    if bound == "memory":
        lines.append(
            f"[roofline] memory-bound (AI={ai:.0f} < ridge {ridge:.0f}). "
            f"HBM util {bw:.1f}%. Increase BLOCK_M/BLOCK_N so tiles reuse data in SRAM; "
            f"raise num_stages to overlap HBM loads with tensor cores."
        )
    else:
        lines.append(
            f"[roofline] compute-bound (AI={ai:.0f} > ridge {ridge:.0f}). "
            f"Compute {eff:.1f}% of 312 TFLOPS. "
            f"Ensure tl.dot operands are multiples of 16; try num_warps=8 if not set."
        )
    if bw < 30 and bound == "memory":
        lines.append(f"[warn] only {bw:.1f}% of HBM bandwidth -- tiles likely too small. Double BLOCK_M or BLOCK_N.")

    # ---- Layer 2: compile metadata -> occupancy ----
    km = result.get("kernel_metadata") or []
    occ = result.get("occupancy") or {}
    if km:
        hot = max(km, key=lambda k: (k.get("n_regs") or 0))
        regs   = hot.get("n_regs") or 0
        spills = hot.get("n_spills") or 0
        shared = hot.get("shared") or 0
        if spills and spills > 0:
            lines.append(
                f"[compile] CRITICAL: {spills} register spills -- kernel is spilling to local memory "
                f"(thrashing L1). Reduce BLOCK_M (try 64) or reduce live values in the inner loop."
            )
        if regs > 128:
            lines.append(
                f"[compile] high register pressure: {regs} regs/thread. A100 has 65536 regs/SM; "
                f"this caps occupancy at ~{occ.get('blocks_per_sm', '?')} blocks/SM. "
                f"Consider BLOCK_M=64 to drop register count."
            )
        if shared > 48 * 1024:
            lines.append(
                f"[compile] shared memory {shared} bytes -- above 48 KB threshold; forces 1 block/SM. "
                f"Try smaller BLOCK_N or reduce num_stages to shrink shared usage."
            )
    if occ.get("blocks_per_sm") is not None:
        bps = occ["blocks_per_sm"]
        limiter = occ.get("limiter", "?")
        occ_pct = occ.get("occupancy_pct", 0)
        if occ_pct < 30:
            lines.append(
                f"[occupancy] LOW: {occ_pct}% ({bps} blocks/SM, limiter={limiter}). "
                f"Latency hiding will suffer. Reduce {limiter} usage."
            )
        elif occ_pct >= 50:
            lines.append(f"[occupancy] good: {occ_pct}% ({bps} blocks/SM).")

    # ---- Layer 3: torch.profiler ----
    tp = result.get("torch_profile") or {}
    dev_ms = tp.get("device_time_ms_per_call")
    wall_ms = result.get("time_ms", 0)
    if dev_ms and wall_ms:
        # do_bench already subtracts most overhead, so dev/wall should be ~1.0
        ratio = dev_ms / wall_ms
        if ratio < 0.7:
            lines.append(
                f"[torch.profiler] device_time={dev_ms:.3f}ms but wall={wall_ms:.3f}ms (ratio {ratio:.2f}). "
                f"Significant host-side overhead -- check launch grid shape."
            )

    # ---- Layer 4: Proton measured vs theoretical ----
    pp = result.get("proton_profile") or {}
    scopes = pp.get("scopes") or []
    if scopes:
        attn = next((s for s in scopes if "attention" in s.get("name", "").lower()), scopes[0])
        measured_bytes = attn.get("bytes") or 0
        theoretical_bytes = 4 * result.get("config", {}).get("batch", 1) * \
                            result.get("config", {}).get("n_heads", 32) * \
                            result.get("config", {}).get("seq_len", 2048) * \
                            result.get("config", {}).get("d_head", 128) * 2
        if measured_bytes and theoretical_bytes:
            inflation = measured_bytes / theoretical_bytes
            if inflation > 1.5:
                lines.append(
                    f"[proton] HBM traffic {inflation:.1f}x theoretical minimum -- "
                    f"K/V being re-read across blocks. Fuse causal tile-skip OR enlarge BLOCK_M."
                )

    if eff < 5:
        lines.append(
            "[critical] under 5% compute efficiency -- kernel may be falling back to scalar ops. "
            "Ensure BLOCK_M, BLOCK_N, HEAD_DIM are all multiples of 16 so tl.dot hits tensor cores."
        )

    return " ".join(lines)


def _build_feedback_message(iteration: int, result: dict, prev_code: str) -> str:
    rules = (
        "Rules reminder: single ```python block, no import statements, "
        "ASCII-only strings/comments, num_stages only in grid launch kwargs, "
        "NO tl.ones() (use tl.full([N], 1.0, dtype=tl.float32) instead), "
        "keep the kernel under 130 lines. "
        "Do NOT use @triton.autotune. "
        "Declare BLOCK_M, BLOCK_N as tl.constexpr in the kernel signature; "
        "pass them plus num_warps/num_stages in the grid launch: "
        "_attn_fwd[grid](..., BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2). "
        "NEVER declare num_warps or num_stages in the @triton.jit signature -- "
        "they are JIT meta-params consumed by the runtime (causes 'missing argument: num_warps'). "
        "Use plain range() not tl.range()."
    )
    if result.get("success"):
        diagnosis = _diagnose(result)
        return (
            f"Iteration {iteration} result: {_fmt_metrics(result)}\n\n"
            f"Diagnosis: {diagnosis}\n\n"
            f"Output the next improved kernel. {rules}"
        )
    else:
        short_err = result.get("error", "unknown error")[:400]
        return (
            f"Iteration {iteration} FAILED: {short_err}\n\n"
            f"Fix the error and output the corrected kernel. {rules}"
        )


def run_optimization_agent(
    kernel_code: str,
    config: dict = None,
    max_iterations: int = MAX_ITERATIONS,
) -> Generator[dict, None, None]:
    if config is None:
        config = DEFAULT_CONFIG

    client = _make_client()

    # --- Step 0: baseline profile ---
    yield {"type": "thought", "text": "Profiling baseline kernel on A100..."}

    baseline = _call_profile_kernel(kernel_code, config)
    if not baseline.get("success"):
        yield {"type": "error", "text": f"Baseline profiling failed: {baseline.get('error')}"}
        return

    yield {"type": "metrics", "data": baseline, "iteration": 0}
    yield {"type": "kernel",  "code": kernel_code, "iteration": 0}

    best_code    = kernel_code
    best_metrics = baseline

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_initial_message(kernel_code, baseline, config)},
    ]

    iteration = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    while iteration < max_iterations:
        # --- Ask LLM for code — retry up to 3x on rate-limit (60s back-off) ---
        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    max_tokens=3000,   # kernel ~200 lines ~1300 tokens; 3000 handles verbose formatting
                    # Qwen3 thinking mode disabled: avoids 500-2000 hidden reasoning tokens
                    extra_body={"thinking": {"type": "disabled"}} if "qwen" in LLM_MODEL.lower() else {},
                )
                break
            except Exception as e:
                err = str(e)
                # Retry on real rate-limit (429) OR transient overload (503).
                is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
                is_overloaded = "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower()
                if (is_rate_limit or is_overloaded) and attempt < 2:
                    wait = 30 * (attempt + 1) if is_overloaded else 60 * (attempt + 1)
                    kind = "overloaded (503)" if is_overloaded else "rate limit (429)"
                    yield {
                        "type": "thought",
                        "text": (
                            f"{LLM_PROVIDER.title()} {kind} — "
                            f"waiting {wait}s, retry {attempt + 1}/3."
                        ),
                    }
                    time.sleep(wait)
                elif is_rate_limit or is_overloaded:
                    yield {
                        "type": "error",
                        "text": (
                            f"{LLM_PROVIDER.title()} rate limit hit after 3 attempts. "
                            f"Raw error: {err[:500]}"
                        ),
                    }
                    return
                else:
                    yield {"type": "error", "text": f"{LLM_PROVIDER.title()} API error: {e}"}
                    return
        if response is None:
            return

        reply = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})

        # Show reasoning to UI (everything outside the code block)
        thought = re.sub(r"```python.*?```", "", reply, flags=re.DOTALL).strip()
        if thought:
            yield {"type": "thought", "text": thought}

        # --- Extract code ---
        new_code = _extract_code_block(reply)

        if not new_code or "attention_kernel" not in new_code:
            yield {"type": "thought", "text": "No valid kernel found in response — asking again."}
            messages.append({
                "role":    "user",
                "content": (
                    "Your response did not contain a valid kernel. "
                    "Output ONLY the complete Python source in a single ```python block "
                    "with the function attention_kernel(q, k, v, is_causal=False, scale=None)."
                ),
            })
            continue   # retry same iteration, don't increment

        # --- Profile it on A100 ---
        yield {"type": "tool_call", "name": "profile_kernel", "input": {"kernel_code": new_code[:200] + "..."}}

        result = _call_profile_kernel(new_code, config)
        iteration += 1

        if result.get("success"):
            consecutive_failures = 0
            yield {"type": "metrics", "data": result, "iteration": iteration}
            yield {"type": "kernel",  "code": new_code, "iteration": iteration}
            if result["tflops"] > best_metrics.get("tflops", 0):
                best_code    = new_code
                best_metrics = result
        else:
            consecutive_failures += 1
            yield {"type": "error", "text": f"Kernel failed: {result.get('error')}"}
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                yield {
                    "type": "thought",
                    "text": (
                        f"Agent stopped: {consecutive_failures} consecutive kernel failures. "
                        "The LLM is stuck in an error loop. Best kernel so far is the baseline."
                    ),
                }
                break

        # Prune old kernel source from assistant messages to stay under TPM limits.
        # Groq free tier: 6000 TPM. A single kernel is ~700 tokens; history grows fast.
        # Keep metrics/diagnosis (small) but replace previous code blocks with a stub.
        for msg in messages:
            if msg["role"] == "assistant" and "```python" in msg["content"]:
                msg["content"] = re.sub(
                    r"```python.*?```",
                    "```python\n# [previous kernel omitted to save tokens]\n```",
                    msg["content"],
                    flags=re.DOTALL,
                )

        # Feed result back so the LLM can improve
        messages.append({
            "role":    "user",
            "content": _build_feedback_message(iteration, result, new_code),
        })

        # Only stop early if we produced at least one kernel that actually
        # beats the baseline AND clears the user's efficiency target.
        # (best_metrics starts as the baseline itself, so iteration > 0 guard
        # prevents the baseline's own efficiency from triggering an early exit.)
        if (
            iteration > 0
            and result.get("success")
            and best_metrics.get("tflops", 0) > baseline.get("tflops", 0)
            and best_metrics.get("efficiency_pct", 0) >= EFFICIENCY_TARGET_PCT
        ):
            yield {
                "type": "thought",
                "text": (
                    f"Target reached: {best_metrics['efficiency_pct']:.1f}% efficiency "
                    f"({best_metrics['tflops']:.2f} TFLOPS). Stopping."
                ),
            }
            break

    # --- Final summary ---
    speedup = best_metrics.get("tflops", 0) / max(baseline.get("tflops", 1e-6), 1e-6)

    _log_motus_trace({
        "timestamp":        time.time(),
        "bound":            baseline.get("bound", "unknown"),
        "baseline_tflops":  baseline.get("tflops", 0),
        "best_tflops":      best_metrics.get("tflops", 0),
        "speedup":          round(speedup, 2),
        "iterations":       iteration,
        "seq_len":          config.get("seq_len"),
        "d_head":           config.get("d_head"),
        "block_m":          _extract_param(best_code, "BLOCK_M"),
        "block_n":          _extract_param(best_code, "BLOCK_N"),
        "num_warps":        _extract_param(best_code, "num_warps"),
        "winning_strategy": (
            "tiling+pipelining" if "num_stages" in best_code
            else "tiling" if int(_extract_param(best_code, "BLOCK_M") or 0) > 64
            else "block_tuning"
        ),
    })

    yield {
        "type":         "done",
        "best_code":    best_code,
        "best_metrics": best_metrics,
        "baseline":     baseline,
        "iterations":   iteration,
        "speedup":      round(speedup, 2),
    }
