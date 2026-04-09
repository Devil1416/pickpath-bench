---
title: PickPath-Bench
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<div align="center">

```
██████╗ ██╗ ██████╗██╗  ██╗██████╗  █████╗ ████████╗██╗  ██╗
██╔══██╗██║██╔════╝██║ ██╔╝██╔══██╗██╔══██╗╚══██╔══╝██║  ██║
██████╔╝██║██║     █████╔╝ ██████╔╝███████║   ██║   ███████║
██╔═══╝ ██║██║     ██╔═██╗ ██╔═══╝ ██╔══██║   ██║   ██╔══██║
██║     ██║╚██████╗██║  ██╗██║     ██║  ██║   ██║   ██║  ██║
╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
                    B  E  N  C  H
```

**A deterministic OpenEnv benchmark for evaluating multi-stop route planning agents**  
*in constrained warehouse logistics environments*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-00e5a0?style=flat-square)](https://github.com/openenv)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063?style=flat-square)](https://docs.pydantic.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## What is PickPath-Bench?

**PickPath-Bench** is a reproducible benchmark environment for evaluating how well an AI agent plans multi-stop pick routes in a constrained warehouse grid. Every movement costs time. Walls create forced detours. Corridors collapse mid-episode. The agent must collect all items as efficiently as possible.

This is not a toy problem. Warehouse route planning — specifically the **Picker Routing Problem (PRP)** — is a well-studied operations research challenge. Every fulfilment centre from Amazon to local 3PLs runs some variant of this optimisation daily. Inefficient pick paths directly translate to higher labour costs, slower order throughput, and floor congestion. PickPath-Bench gives the RL and agent community a clean, reproducible environment to measure planning capability against a ground-truth BFS-optimal baseline.

```
Observed gap in the field
──────────────────────────────────────────────────────────
Human pickers:      ~65–80% of theoretical optimal
Rule-based systems: ~80–90% of theoretical optimal
Frontier LLMs:      ???  ← this is what PickPath-Bench measures
```

---

## Environment Overview

```
┌─────────────────────────────────────────────────────┐
│  Agent  ◆  navigates a discrete N×N grid            │
│  Items  ▪  must all be collected (order matters)    │
│  Walls  █  static obstacles blocking direct paths   │
│  Events ⚡  corridors that close at a specific step  │
└─────────────────────────────────────────────────────┘
```

The core insight: the agent selects targets by **Manhattan distance** (straight-line), but the actual cost is **BFS distance** (wall-aware shortest path). When a wall sits between the agent and the nearest-looking item, the agent pays a heavy detour penalty — unless it plans globally instead of greedily.

---

## API

PickPath-Bench implements the full OpenEnv interface:

```python
from env.env import WarehouseBotEnv

env = WarehouseBotEnv(task_id="medium")
obs = env.reset("medium")       # → ObservationModel
result = env.step("right")      # → RewardModel  (obs, reward, done, info)
state = env.state()             # → ObservationModel (no side effects)
```

All inputs and outputs are **typed Pydantic v2 models** — no raw dicts, no implicit state.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `grid_size` | `int` | Side length of the square grid |
| `agent_position` | `GridPosition` | Current `(row, col)` of the agent |
| `item_positions` | `list[GridPosition]` | Remaining uncollected items |
| `picked_items` | `list[GridPosition]` | Items collected this episode |
| `obstacles` | `list[GridPosition]` | All impassable cells (static + triggered dynamic) |
| `step_count` | `int` | Steps elapsed this episode |
| `task_id` | `str` | Active task identifier |
| `done` | `bool` | `True` when complete or step budget exhausted |
| `total_items` | `int` | Total items to collect |
| `available_actions` | `list[str]` | Always `["up","down","left","right"]` |

## Action Space

Discrete, 4-connected movement:

| Action | Effect |
|---|---|
| `up` | Move agent to `(row−1, col)` |
| `down` | Move agent to `(row+1, col)` |
| `left` | Move agent to `(row, col−1)` |
| `right` | Move agent to `(row, col+1)` |

Invalid moves (wall / out-of-bounds) are **penalised** but do not terminate the episode.

---

## Reward Function

Dense per-step rewards provide continuous learning signal throughout the episode:

```
reward = −1          every step             (efficiency pressure)
       + 50          on item collection     (task progress signal)
       − 10          on invalid move        (collision penalty)
       + 0–10        on episode completion  (early-finish bonus)
```

The early-finish bonus scales as `10 × (steps_remaining / max_steps)`, rewarding agents that finish fast, not just agents that finish.

**Episode termination:** all items collected, OR `max_steps` reached (partial completion scored).

---

## Tasks

### `easy` — Open Floor, Linear Layout (5×5, 3 items)

```
S . A . .
. . . . .
. . B . .
. . . . .
. . . . C
```

Items lie on a roughly diagonal path. Greedy nearest-first selection produces the globally optimal order. Purpose: verify the grader, establish a near-perfect baseline below the validator ceiling, and confirm an LLM can follow basic navigation instructions.

| Metric | Value |
|---|---|
| Grid | 5 × 5 |
| Items | 3 |
| Obstacles | 0 |
| Max steps | 50 |
| Optimal steps | 8 |
| **Baseline score** | **0.9900** |

---

### `medium` — Down-Wall Bait Trap (6×6, 4 items)

```
S . . . D .
# # . . . .   ← wall blocks (1,0) and (1,1)
A . . . . .
. . . . . .
. . . . C .
. B . . . .
```

Item **A** at `(2,0)` has Manhattan distance 2 — the closest item from start. But the wall forces a 6-step BFS detour to reach it. An agent that commits to A first then backtracks across the grid to collect B, C, and D in a suboptimal order. The optimal agent ignores the Manhattan bait, collects D first (a free rightward sweep), then A, B, C in sequence.

```
Greedy (Manhattan-first): S → A(6) → B(3) → C(5) → D(4) = 18 steps
Optimal:                  S → D(4) → A(6) → B(2) → C(4) = 16 steps
```

| Metric | Value |
|---|---|
| Grid | 6 × 6 |
| Items | 4 |
| Static obstacles | 2 |
| Max steps | 60 |
| Optimal steps | 16 |
| **Baseline score** | **0.8889** |

---

### `hard` — Wall + Corridor Collapse (7×7, 5 items)

```
S . # B . . .
. . # . . . .
. . # . . . G
. . # . . . .
A . · . . . .   ← gap (4,2) open initially, closes at step 4
. . . . . C .
. . E . . . .
```

A static 4-cell wall at `col=2, rows 0–3` blocks the direct path to the right side of the grid. Item **B** at `(0,3)` appears closest by Manhattan (distance 3) but requires an 11-step BFS detour — the agent must go all the way down to row 4 to cross the wall, then back up. At **step 4**, a dynamic event closes the crossing at `(4,2)`, forcing any agent that hasn't crossed yet to detour further via row 5 or 6.

An agent that chases B first gets stranded in the top-right with items A and E still in the bottom-left, triggering a massive backtrack. The optimal agent ignores the bait entirely, sweeps the open left-bottom side first, then crosses once to collect the right-side items.

```
Greedy (bait-first): S→B(11)→G(6)→C(4)→E(9)→A(5)  ≈ 28 steps
Optimal:             S→A(4)→E(4)→C(6)→G(5)→B(6)   = 21 steps
```

| Metric | Value |
|---|---|
| Grid | 7 × 7 |
| Items | 5 |
| Static obstacles | 4 |
| Dynamic events | 1 (step 4) |
| Max steps | 80 |
| Optimal steps | 21 |
| **Baseline score** | **0.7000** |

---

## Grader

The grader computes the **true BFS-optimal route** by exhaustively evaluating all item orderings and comparing against the agent's actual step count:

```python
score = (optimal_steps / actual_steps) × (items_collected / total_items)
```

Clamped to `[0.01, 0.99]`. An optimal run reaches the upper ceiling instead of 1.0 exactly so validators that round scores never push it out of range. The grader uses real BFS distances — not Manhattan approximations — making it strictly harder to fool than the baseline agent.

---

## Baseline Scores

Scores produced by the deterministic planner in `inference.py` (no LLM):

| Task | Optimal Steps | Agent Steps | Score |
|---|---|---|---|
| Easy | 8 | 8 | **0.9900** |
| Medium | 16 | 18 | **0.8889** |
| Hard | 21 | 30 | **0.7000** |
| **Overall** | — | — | **0.8596** |

---

## Inference Script

`inference.py` runs all three tasks end-to-end. When `HF_TOKEN` is set, it queries the LLM via the OpenAI-compatible client. When absent, it falls back to the deterministic planner — so the script **always runs to completion** regardless of API availability.

**Output format** (strictly followed):
```
[START] task=easy env=pickpath-bench model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=right reward=-1.00 done=false error=null
[STEP] step=2 action=right reward=49.00 done=false error=null
...
[END] success=true steps=8 score=0.990000 rewards=-1.00,-1.00,49.00,...
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | *(none)* | Hugging Face / API key |

---

## Setup

**Local:**
```bash
git clone https://huggingface.co/spaces/harshnotfound/warehouse-bot-env
cd warehouse-bot-env
pip install -r requirements.txt
python inference.py
```

**With LLM:**
```bash
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

**Docker:**
```bash
docker build -t pickpath-bench .
docker run --rm pickpath-bench

# With LLM
docker run --rm \
  -e HF_TOKEN=hf_... \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  pickpath-bench
```

---

## Project Structure

```
pickpath-bench/
├── env/
│   ├── __init__.py
│   ├── env.py          # WarehouseBotEnv — reset / step / state
│   ├── models.py       # Pydantic v2 typed models
│   ├── tasks.py        # Task definitions + dynamic obstacle events
│   └── graders.py      # BFS exhaustive optimal-path grader
├── inference.py        # Baseline runner — OpenAI client + planner fallback
├── app.py              # Flask web UI with live grid visualisation (port 7860)
├── openenv.yaml        # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Why This Benchmark Is Hard to Game

Most grid navigation benchmarks fail because the optimal policy is trivially learnable. PickPath-Bench is designed so that **locally rational behaviour is globally suboptimal**:

1. **Manhattan ≠ BFS** — walls make the nearest-looking item expensive to reach
2. **Committed routing** — the agent locks a target on selection and cannot greedily redirect mid-path
3. **Dynamic events** — corridors close mid-episode, invalidating routes planned at reset
4. **Exhaustive grader** — ground-truth optimal is computed by full BFS permutation search, not approximated

An agent must reason about global item ordering — not just local distance — to score above the baseline.

---

<div align="center">

*Built for the OpenEnv Hackathon · Warehouse logistics × agent planning evaluation*

</div>
