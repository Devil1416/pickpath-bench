from __future__ import annotations

import os

from env.env import WarehouseBotEnv
from env.graders import grade_episode, optimal_steps_for_task
from env.models import ActionType, ObservationModel
from env.tasks import get_task, list_tasks

ACTION_DELTAS: dict[ActionType, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Deterministic tiebreaker for moves of equal distance
ACTION_ORDER: tuple[ActionType, ...] = ("right", "down", "left", "up")


def _manhattan_distance(start: tuple[int, int], end: tuple[int, int]) -> int:
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def select_target(
    current_position: tuple[int, int],
    remaining_items: list[tuple[int, int]],
) -> tuple[int, int]:
    """
    Pick the nearest item by Manhattan distance.
    Ties broken deterministically by (row, col).

    ⚠️  Manhattan ignores walls. This is the primary source of suboptimality:
    an item that looks close in straight-line distance may require a costly
    wall detour to actually reach.
    """
    return min(
        remaining_items,
        key=lambda pos: (_manhattan_distance(current_position, pos), pos[0], pos[1]),
    )


def choose_action(
    observation: ObservationModel,
    committed_target: tuple[int, int],
    visit_counts: dict[tuple[int, int], int],
    previous_position: tuple[int, int] | None,
) -> ActionType:
    """
    Move one step toward the committed target.

    The agent does NOT re-evaluate which item to pursue mid-journey.
    It was handed a target by the caller and follows through to completion.
    This commitment is the key behavioural limitation:

    - Target was selected by Manhattan distance (ignoring walls).
    - If that target requires a wall detour, the agent pays the full cost
      without reconsidering whether a different item would have been cheaper.

    Move sorting: distance-to-target + mild revisit penalty (0.15).
    The penalty prevents trivial 2-step oscillation but is intentionally
    too weak to redirect the agent toward a globally better target.
    """
    current_position = (
        observation.agent_position.row,
        observation.agent_position.col,
    )

    obstacles = {(obs.row, obs.col) for obs in observation.obstacles}

    valid_moves: list[tuple[ActionType, tuple[int, int], int]] = []
    for action in ACTION_ORDER:
        dr, dc = ACTION_DELTAS[action]
        next_pos = (current_position[0] + dr, current_position[1] + dc)
        if not (0 <= next_pos[0] < observation.grid_size
                and 0 <= next_pos[1] < observation.grid_size):
            continue
        if next_pos in obstacles:
            continue
        dist = _manhattan_distance(next_pos, committed_target)
        valid_moves.append((action, next_pos, dist))

    if not valid_moves:
        return "up"

    # Avoid immediate backtracking (prevents trivial 2-step oscillation only)
    non_backtrack = [m for m in valid_moves if m[1] != previous_position]
    candidates = non_backtrack or valid_moves

    candidates.sort(
        key=lambda m: (
            m[2] + visit_counts.get(m[1], 0) * 0.15,
            ACTION_ORDER.index(m[0]),
        )
    )
    return candidates[0][0]


def run_task(task_id: str) -> float:
    _api_base_url = os.getenv("API_BASE_URL", "")
    _model_name = os.getenv("MODEL_NAME", "")
    _hf_token = os.getenv("HF_TOKEN", "")

    task = get_task(task_id)
    env = WarehouseBotEnv(task_id=task_id)
    observation = env.reset(task_id)
    optimal_steps = optimal_steps_for_task(task_id)

    previous_position: tuple[int, int] | None = None
    visit_counts: dict[tuple[int, int], int] = {
        (observation.agent_position.row, observation.agent_position.col): 1
    }

    # Select initial target once. Target is locked until it is collected,
    # then a new one is selected from the remaining items.
    remaining = sorted(
        (item.row, item.col) for item in observation.item_positions
    )
    committed_target = select_target(
        (observation.agent_position.row, observation.agent_position.col),
        remaining,
    )

    print("[START]")
    print(f"Task: {task.name}")

    while not observation.done:
        action = choose_action(
            observation, committed_target, visit_counts, previous_position
        )

        current_position = (
            observation.agent_position.row,
            observation.agent_position.col,
        )

        result = env.step(action)
        observation = result.observation

        next_position = (
            observation.agent_position.row,
            observation.agent_position.col,
        )

        previous_position = current_position
        visit_counts[next_position] = visit_counts.get(next_position, 0) + 1

        print(f"[STEP] action={action} reward={result.reward:.1f}")

        # If the target was just collected, pick the next one
        if result.info.item_collected:
            remaining = sorted(
                (item.row, item.col) for item in observation.item_positions
            )
            if remaining:
                committed_target = select_target(next_position, remaining)

        if result.done:
            break

    actual_steps = observation.step_count
    score = grade_episode(
        task_id=task_id,
        actual_steps=actual_steps,
        items_collected=len(observation.picked_items),
    )
    efficiency = (optimal_steps / actual_steps) if actual_steps > 0 else 0.0

    print("[END]")
    print(f"Score: {score:.4f}")
    print(f"Steps: {actual_steps}")
    print(f"Optimal: {optimal_steps}")
    print(f"Efficiency: {efficiency:.2f}")
    print()

    return score


def main() -> None:
    scores: dict[str, float] = {}

    for task in list_tasks():
        scores[task.name] = run_task(task.task_id)

    overall = (sum(scores.values()) / len(scores)) if scores else 0.0

    print("Final Results:")
    print(f"Easy:    {scores['Easy']:.4f}")
    print(f"Medium:  {scores['Medium']:.4f}")
    print(f"Hard:    {scores['Hard']:.4f}")
    print(f"Overall: {overall:.4f}")


if __name__ == "__main__":
    main()