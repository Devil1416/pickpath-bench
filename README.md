<<<<<<< HEAD
---
title: Warehouse Bot Env
emoji: 🐨
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
# Warehouse Bot Env

`warehouse-bot-env` is a deterministic warehouse picking optimisation environment built as a complete OpenEnv-style submission. An agent navigates a grid warehouse, collects all required items, and is scored on how closely its route matches the BFS-optimal path.

## Problem Description

Warehouse picking is a real logistics challenge: every unnecessary movement increases fulfillment latency, labour cost, and floor congestion. This environment models a simplified picking workflow where an agent must collect all items while minimising travel steps and reacting to constrained aisles, including corridors that close mid-episode.

## Real-World Relevance

The environment captures core warehouse behaviours that appear in real fulfillment centres:

- pick-path route planning under physical constraints
- blocked aisles and dynamic corridor changes
- efficiency-driven task execution benchmarking
- deterministic, reproducible policy evaluation

## Environment Design

The environment lives in `env/` and exposes three core methods:

- `reset(task_id)` — initialises a fixed task layout and returns the starting observation
- `step(action)` — applies one movement action and returns a typed `RewardModel`
- `state()` — returns the current `ObservationModel` without advancing the episode

### Observation

| Field | Type | Description |
|---|---|---|
| `grid_size` | int | Side length of the square grid |
| `agent_position` | GridPosition | Current row/col of the agent |
| `item_positions` | list[GridPosition] | Remaining uncollected items |
| `picked_items` | list[GridPosition] | Items collected this episode |
| `obstacles` | list[GridPosition] | All impassable cells (static + dynamic triggered so far) |
| `step_count` | int | Steps taken so far |
| `task_id` | str | Active task identifier |
| `done` | bool | True when complete or step limit reached |
| `total_items` | int | Total items in this task |

### Actions

The action space is discrete: `up`, `down`, `left`, `right`.

### Typed Models

All I/O uses Pydantic v2 models:

- `ObservationModel` — full environment state
- `ActionModel` — single movement action
- `RewardModel` — step result (reward, done, observation, info)
- `StepInfoModel` — per-step metadata

## Tasks

### Easy (5×5, 3 items, no obstacles)

```
S . A . .
. . . . .
. . B . .
. . . . .
. . . . C
```

The greedy nearest-first order is also the optimal order. A perfect agent scores **1.0**. Purpose: establish a clean baseline and verify the grader.

### Medium (6×6, 4 items, static wall)

```
S . . . D .
# # . . . .   ← wall blocks direct path down-left
A . . . . .
. . . . . .
. . . . C .
. B . . . .
```

Item A at (2,0) looks nearest by Manhattan distance but requires a 6-step wall detour. An agent that chases A first will backtrack heavily. Optimal order: D→A→B→C = **16 steps**. Baseline planner takes 18 steps → score ≈ **0.889**.

### Hard (7×7, 5 items, static wall + dynamic corridor closure)

```
S . # B . . .
. . # . . . .
. . # . . . G
. . # . . . .
A . . . . . .   ← gap at (4,2) closes at step 4
. . . . . C .
. . E . . . .
```

A static wall at col=2 rows 0–3 blocks direct right-crossing. At **step 4** the low corridor at (4,2) closes, extending any late-crossing agent's detour by 2+ extra steps. Item B at (0,3) is nearest by Manhattan but costs 11 BFS steps to reach. Optimal order: A→E→C→G→B = **~25 steps**. Baseline planner takes ~30 steps → score ≈ **0.833**.

## Reward Function

Rewards are provided on every step for dense feedback:

| Event | Reward |
|---|---|
| Each step taken | −1 |
| Item collected | +50 |
| Invalid move (wall / out of bounds) | −10 |
| Episode finish bonus (proportional to unused step budget) | up to +10 |

The finish bonus is `10 × (steps_remaining / max_steps)`, encouraging faster completion rather than just completion.

## Grader

The grader uses BFS (not Manhattan distance) to find the true optimal path through all orderings:

```
score = (optimal_steps / actual_steps) × (items_collected / total_items)
```

Clamped to `[0.0, 1.0]`. The grader deliberately uses `itertools.permutations` over all item orderings, making it smarter than the Manhattan-greedy baseline agent — which is the point.

## Baseline Scores

| Task | Score |
|---|---|
| Easy | 1.0000 |
| Medium | ≈ 0.8889 |
| Hard | ≈ 0.8333 |
| **Overall** | **≈ 0.9074** |

## Inference Script

`inference.py` runs all three tasks end-to-end and emits structured logs in the required format:

```
[START] task=easy env=warehouse-bot-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=right reward=-1.00 done=false error=null
...
[END] success=true steps=8 score=1.00 rewards=-1.00,...,49.00
```

When `HF_TOKEN` is set the script uses the OpenAI-compatible client to query `API_BASE_URL` with `MODEL_NAME`. If the token is absent it falls back to the deterministic planner, so the script always runs to completion.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | *(none)* | Hugging Face / API key |

## Local Run

```bash
pip install -r requirements.txt
python inference.py
```

## Docker Run

```bash
docker build -t warehouse-bot-env .
docker run --rm warehouse-bot-env
```

The container runs `inference.py` by default. Pass env vars for LLM mode:

```bash
docker run --rm \
  -e HF_TOKEN=hf_... \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  warehouse-bot-env
```

## File Layout

```
warehouse-bot-env/
├── env/
│   ├── __init__.py
│   ├── env.py        # WarehouseBotEnv — reset/step/state
│   ├── models.py     # Pydantic typed models
│   ├── tasks.py      # Task definitions + dynamic events
│   └── graders.py    # BFS-based optimal-path grader
├── inference.py      # Baseline runner (OpenAI client + planner fallback)
├── openenv.yaml      # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```
>>>>>>> b6d7def9ea967c93a4daf2dd6c4d495a2f583078
