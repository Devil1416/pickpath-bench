from __future__ import annotations

import os

import numpy as np

from env.env import WarehouseBotEnv
from env.graders import grade_episode, optimal_plan_from_observation
from env.tasks import get_task, list_tasks


def run_task(task_id: str) -> float:
    _api_base_url = os.getenv("API_BASE_URL", "")
    _model_name = os.getenv("MODEL_NAME", "")
    _hf_token = os.getenv("HF_TOKEN", "")

    task = get_task(task_id)
    env = WarehouseBotEnv(task_id=task_id)
    observation = env.reset(task_id)

    print("[START]")
    print(f"Task: {task.name}")

    while not observation.done:
        plan = optimal_plan_from_observation(task, observation)
        action = plan[0] if plan else "up"
        result = env.step(action)
        observation = result.observation
        print(f"[STEP] action={action} reward={result.reward:.1f}")

        if result.done:
            break

    score = grade_episode(
        task_id=task_id,
        actual_steps=observation.step_count,
        items_collected=len(observation.picked_items),
    )
    print("[END]")
    print(f"Score: {score:.4f}")
    print()
    return score


def main() -> None:
    scores: dict[str, float] = {}
    for task in list_tasks():
        scores[task.name] = run_task(task.task_id)

    overall = float(np.mean(list(scores.values()))) if scores else 0.0
    print("Final Results:")
    print(f"Easy: {scores['Easy']:.4f}")
    print(f"Medium: {scores['Medium']:.4f}")
    print(f"Hard: {scores['Hard']:.4f}")
    print(f"Overall: {overall:.4f}")


if __name__ == "__main__":
    main()

