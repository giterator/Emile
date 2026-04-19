# GPU Kernel Optimization Agent

Autonomous AI agent that iteratively rewrites a Triton attention kernel to maximize
hardware utilization on an A100-80GB, then hooks the optimized kernel into Qwen3-4B
to show real inference speedup — live, in a side-by-side streaming UI.

## Architecture

```
Streamlit UI
    │
    ├─ Tab 1: Agent Optimizer
    │     Motus harness  ◄──────────────────────────────────────────┐
    │          │   (retrieves past winning strategies as context)    │
    │          ▼                                                     │
    │     Claude 3.5 Sonnet (tool-calling loop)                     │
    │          └─ profile_kernel() ──► Modal A100                   │
    │                                      Triton bench → metrics   │
    │          optimization trace logged ──────────────────────────►┘
    │          Motus learns: bound type → strategy → speedup
    │
    └─ Tab 2: Live Demo
          run_inference_comparison() ──► Modal A100
               Qwen3-4B baseline  (PyTorch SDPA)
               Qwen3-4B optimized (Triton kernel via SDPA hook)
               Streaming tokens → side-by-side UI
```

## How Motus improves the agent over time

Motus observes every optimization trace:
- Which bottleneck type (memory-bound / compute-bound)
- Which strategy was applied (block size, warp count, pipelining)
- What speedup resulted
- How many iterations it took to converge

On the next run, Motus injects the retrieved best strategies as context before the
agent starts. The agent converges in fewer iterations. This is the "learns in prod"
flywheel — demonstrated live in a hackathon by running the optimizer twice on similar
kernels and showing the second run reaching target in fewer steps.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free Gemini API key

The agent uses **Gemini 2.0 Flash** — free, no credit card needed.

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API key** → Create API key
3. Copy it into your `.env`:

```bash
cp .env.example .env
# set GOOGLE_API_KEY=AIza...
```

### 3. Modal setup

```bash
modal setup          # authenticate with your Modal account
```

Qwen3-4B is a public model — no HuggingFace token needed.

### 4. Deploy Modal functions

```bash
modal deploy modal_app.py
```

### 5. Install and deploy with Motus

[Motus](https://www.lithosai.com/) wraps the agent as its serving harness and learns
from every optimization run — so each subsequent kernel starts with retrieved context
from past successes instead of cold-starting.

```bash
# Install the Motus CLI
curl -fsSL https://www.lithosai.com/motus/install.sh | sh

# Deploy the agent (cloud-hosted, no Dockerfiles or K8s config needed)
motus deploy
```

Or self-host:

```bash
motus serve
```

### 6. Run the UI

```bash
streamlit run ui/app.py
```

## Project Structure

```
kernel-optimizer/
├── modal_app.py          # A100 GPU functions (profiling + inference)
├── agent.py              # Claude tool-calling optimization loop
├── hook.py               # SDPA monkey-patch for Qwen3-4B
├── kernels/
│   ├── reference.py        # PyTorch correctness oracle
│   ├── fa2_official.py     # Baseline: Triton's official FA2 (A100 pointer path)
│   └── v1_naive_triton.py  # Archived naive kernel (kept for reference)
├── prompts/
│   └── optimizer.py      # System prompt + tool schemas
├── ui/
│   └── app.py            # Streamlit UI
└── requirements.txt
```

## How the agent works

1. **Baseline profile** — Official Triton FA2 kernel (A100 pointer path) profiled on A100
2. **Diagnosis** — LLM reads TFLOPS, bandwidth, arithmetic intensity, bound type
3. **Rewrite** — LLM generates improved kernel (better block tiles, warps, pipeline depth)
4. **Verify** — Modal execs the new code, checks correctness vs PyTorch reference
5. **Iterate** — Loop until efficiency target or max iterations hit
6. **Hook** — Best kernel patched into Qwen3-4B via SDPA override for inference demo

## Expected performance

| Kernel | TFLOPS | vs A100 peak | Speedup |
|--------|--------|--------------|---------|
| Triton official FA2 (autotuned) | ~10-18 | ~3-6% | 1× baseline |
| Agent-optimized (tuned for Qwen3 shapes) | ~18-30 | ~6-10% | ~1.5-2× |

*Attention is memory-bound at all practical sequence lengths — the roofline ceiling
for attention is ~20 TFLOPS, not 312. Reaching 20+ TFLOPS matches FlashAttention-2
on A100 and is the hackathon target.*
