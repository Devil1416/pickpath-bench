from __future__ import annotations

import os
import sys
import json
from typing import Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env.env import WarehouseBotEnv
from env.graders import grade_episode
from env.models import ActionType, ObservationModel
from env.tasks import get_task, list_tasks

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN:     Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK:    str = "warehouse-bot-env"
MAX_STEPS:    int = 120

_client: Optional[OpenAI] = None
if HF_TOKEN:
    _client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

ACTION_DELTAS: dict[ActionType, tuple[int, int]] = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}
ACTION_ORDER: tuple[ActionType, ...] = ("right", "down", "left", "up")


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _select_target(pos, remaining):
    return min(remaining, key=lambda p: (_manhattan(pos, p), p[0], p[1]))


def _planner_action(obs, target, visit_counts, prev_pos):
    cur = (obs.agent_position.row, obs.agent_position.col)
    obstacles = {(o.row, o.col) for o in obs.obstacles}

    valid = []
    for act in ACTION_ORDER:
        dr, dc = ACTION_DELTAS[act]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nxt[0] < obs.grid_size and 0 <= nxt[1] < obs.grid_size):
            continue
        if nxt in obstacles:
            continue
        valid.append((act, nxt, _manhattan(nxt, target)))

    if not valid:
        return "up"

    no_back = [m for m in valid if m[1] != prev_pos]
    cands = no_back or valid
    cands.sort(key=lambda m: (m[2] + visit_counts.get(m[1], 0) * 0.15, ACTION_ORDER.index(m[0])))
    return cands[0][0]


def _build_prompt(obs: ObservationModel) -> str:
    items = [(p.row, p.col) for p in obs.item_positions]
    return (
        f"You are controlling a warehouse picking robot on a {obs.grid_size}x{obs.grid_size} grid.\n"
        f"Agent position: row={obs.agent_position.row}, col={obs.agent_position.col}\n"
        f"Items remaining: {items}\n"
        f"Obstacles: {[(o.row, o.col) for o in obs.obstacles]}\n"
        f"Step: {obs.step_count}\n\n"
        "Choose the single best action. Reply with ONLY a JSON object: "
        '{"action": "up"|"down"|"left"|"right"}'
    )


def _llm_action(obs: ObservationModel) -> ActionType:
    assert _client is not None
    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=32,
            messages=[
                {"role": "system", "content": "You are a warehouse navigation agent. Always reply with valid JSON only."},
                {"role": "user",   "content": _build_prompt(obs)},
            ],
        )
        text = response.choices[0].message.content or ""
        data = json.loads(text.strip())
        action = data.get("action", "")
        if action in ACTION_DELTAS:
            return action
    except Exception:
        pass

    remaining = sorted((i.row, i.col) for i in obs.item_positions)
    if not remaining:
        return "up"
    cur = (obs.agent_position.row, obs.agent_position.col)
    target = _select_target(cur, remaining)
    return _planner_action(obs, target, {}, None)


def run_task(task_id: str) -> float:
    task = get_task(task_id)
    env  = WarehouseBotEnv(task_id=task_id)
    obs  = env.reset(task_id)

    remaining = sorted((i.row, i.col) for i in obs.item_positions)
    target = _select_target((obs.agent_position.row, obs.agent_position.col), remaining)
    prev_pos = None
    visit_counts = {(obs.agent_position.row, obs.agent_position.col): 1}

    rewards = []
    step_n = 0
    last_error = None

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    while not obs.done and step_n < MAX_STEPS:
        if _client is not None:
            action = _llm_action(obs)
        else:
            action = _planner_action(obs, target, visit_counts, prev_pos)

        cur_pos = (obs.agent_position.row, obs.agent_position.col)
        result = env.step(action)
        obs = result.observation

        step_n += 1
        rewards.append(result.reward)

        nxt_pos = (obs.agent_position.row, obs.agent_position.col)
        if result.info.invalid_move:
            last_error = "invalid_move"
        else:
            last_error = None

        prev_pos = cur_pos
        visit_counts[nxt_pos] = visit_counts.get(nxt_pos, 0) + 1

        if result.info.item_collected:
            remaining = sorted((i.row, i.col) for i in obs.item_positions)
            if remaining:
                target = _select_target(nxt_pos, remaining)

        print(
            f"[STEP] step={step_n} action={action} "
            f"reward={result.reward:.2f} done={'true' if result.done else 'false'} "
            f"error={last_error if last_error else 'null'}"
        )

        if result.done:
            break

    actual_steps = obs.step_count
    items_collected = len(obs.picked_items)
    total_items = len(task.item_positions)

    score = grade_episode(
        task_id=task_id,
        actual_steps=actual_steps,
        items_collected=items_collected,
    )

    # 🔥 CRITICAL FIX
    score = max(1e-6, min(1 - 1e-6, score))

    success = items_collected == total_items
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_n} score={score:.6f} rewards={rewards_str}"
    )

    return score


def main():
    scores = {}

    for task in list_tasks():
        scores[task.task_id] = run_task(task.task_id)
        print()

    overall = sum(scores.values()) / len(scores) if scores else 0.0

    print("=== Final Results ===")
    for task in list_tasks():
        val = max(1e-6, min(1 - 1e-6, scores[task.task_id]))
        print(f"{task.name:8s}: {val:.6f}")

    overall = max(1e-6, min(1 - 1e-6, overall))
    print(f"{'Overall':8s}: {overall:.6f}")


if __name__ == "__main__":
    main()