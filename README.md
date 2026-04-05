# Warehouse Bot Env

`warehouse-bot-env` is a deterministic warehouse picking optimization environment built as a complete OpenEnv-style hackathon submission. The agent operates in a grid-based warehouse, collects items, and is scored on how efficiently it completes each picking task.

## Problem Description

Warehouse picking is a real logistics problem: each unnecessary move increases fulfillment latency, labor cost, and congestion on the floor. This environment models a simplified picking workflow where an agent must collect all required items while minimizing travel steps and reacting to constrained aisles.

## Real-World Relevance

The environment represents core warehouse behaviors that appear in fulfillment centers:

- route planning for pick paths
- congestion and blocked aisles
- efficiency-driven task execution
- deterministic benchmarking of policy quality

## Environment Design

The environment lives in `env/` and exposes three core methods:

- `reset(task_id)` initializes a fixed task layout
- `step(action)` applies one movement action and returns a typed reward payload
- `state()` returns the current typed observation

### State

Each observation includes:

- `agent_position`
- `item_positions`
- `picked_items`
- `obstacles`
- `step_count`
- `task_id`

### Actions

The action space is discrete and deterministic:

- `up`
- `down`
- `left`
- `right`

### Typed Models

Pydantic models are used for:

- `ObservationModel`
- `ActionModel`
- `RewardModel`

## Tasks

### Easy

- 5x5 grid
- 3 items
- no obstacles
- fixed layout

### Medium

- 6x6 grid
- 4 items
- fixed divider wall
- multiple valid paths around the obstruction

### Hard

- 6x6 grid
- 4 items
- fixed layout
- after step 4, a new obstacle is introduced deterministically
- the agent must replan through a lower aisle

## Reward Logic

Rewards are provided on every step:

- `-1` per step
- `+50` when an item is collected
- `-10` for an invalid move or obstacle collision
- finishing bonus based on unused step budget

This provides dense feedback while still favoring globally efficient completion.

## Deterministic Grading

The grader is deterministic and uses:

`score = (optimal_steps / actual_steps) * (items_collected / total_items)`

The final score is clamped to the range `[0.0, 1.0]`.

There is no randomness in task layouts, event timing, or scoring.

## Inference

`inference.py` runs all three tasks end-to-end and prints deterministic logs in the required format. It also reads these environment variables for compatibility with model-serving workflows:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

The included reference runner uses a deterministic planner so it can execute without external services and without crashing.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference:

```bash
python inference.py
```

## Docker Run

Build the image:

```bash
docker build -t warehouse-bot-env .
```

Run the container:

```bash
docker run --rm warehouse-bot-env
```

The container runs `inference.py` by default.

## Hugging Face Deployment Notes

This project is ready for container-based deployment because:

- dependencies are pinned in `requirements.txt`
- the runtime entrypoint is defined in `Dockerfile`
- execution is deterministic
- inference does not require network access

## File Layout

```text
warehouse-bot-env/
├── env/
│   ├── env.py
│   ├── models.py
│   ├── tasks.py
│   └── graders.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```
