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

ACTION_ORDER: tuple[ActionType, ...] = ("right", "down", "left", "up")


def _manhattan_distance(start: tuple[int, int], end: tuple[int, int]) -> int:
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def _direct_path_blockers(
    start: tuple[int, int],
    end: tuple[int, int],
    obstacles: set[tuple[int, int]],
) -> int:
    def walk_path(
        first_axis: str,
    ) -> list[tuple[int, int]]:
        row, col = start
        target_row, target_col = end
        path: list[tuple[int, int]] = []

        if first_axis == "row":
            row_step = 1 if target_row > row else -1
            while row != target_row:
                row += row_step
                path.append((row, col))

            col_step = 1 if target_col > col else -1
            while col != target_col:
                col += col_step
                path.append((row, col))
        else:
            col_step = 1 if target_col > col else -1
            while col != target_col:
                col += col_step
                path.append((row, col))

            row_step = 1 if target_row > row else -1
            while row != target_row:
                row += row_step
                path.append((row, col))

        return path

    if start == end:
        return 0

    row_first = walk_path("row")
    col_first = walk_path("col")
    row_first_blockers = sum(1 for position in row_first if position in obstacles)
    col_first_blockers = sum(1 for position in col_first if position in obstacles)
    return min(row_first_blockers, col_first_blockers)


def choose_action(
    observation: ObservationModel,
    visit_counts: dict[tuple[int, int], int],
    previous_position: tuple[int, int] | None,
) -> ActionType:
    current_position = (
        observation.agent_position.row,
        observation.agent_position.col,
    )
    remaining_items = sorted(
        (item.row, item.col) for item in observation.item_positions
    )
    if not remaining_items:
        return "up"

    obstacles = {(obstacle.row, obstacle.col) for obstacle in observation.obstacles}

    target = min(
        remaining_items,
        key=lambda item_position: (
            _manhattan_distance(current_position, item_position),
            _direct_path_blockers(current_position, item_position, obstacles),
            item_position[0],
            item_position[1],
        ),
    )
    current_distance = _manhattan_distance(current_position, target)

    valid_moves: list[tuple[ActionType, tuple[int, int], int, int]] = []
    for action in ACTION_ORDER:
        delta_row, delta_col = ACTION_DELTAS[action]
        next_position = (
            current_position[0] + delta_row,
            current_position[1] + delta_col,
        )
        if not (
            0 <= next_position[0] < observation.grid_size
            and 0 <= next_position[1] < observation.grid_size
        ):
            continue
        if next_position in obstacles:
            continue

        next_distance = _manhattan_distance(next_position, target)
        blocker_count = _direct_path_blockers(next_position, target, obstacles)
        valid_moves.append((action, next_position, next_distance, blocker_count))

    if not valid_moves:
        return "up"

    non_backtracking_moves = [
        move for move in valid_moves if move[1] != previous_position
    ]
    candidate_moves = non_backtracking_moves or valid_moves
    candidate_moves.sort(
        key=lambda move: (
            0 if move[2] < current_distance else 1,
            move[3],
            move[2],
            visit_counts.get(move[1], 0),
            ACTION_ORDER.index(move[0]),
        )
    )
    return candidate_moves[0][0]


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

    print("[START]")
    print(f"Task: {task.name}")

    while not observation.done:
        action = choose_action(observation, visit_counts, previous_position)
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
    print(f"Easy: {scores['Easy']:.4f}")
    print(f"Medium: {scores['Medium']:.4f}")
    print(f"Hard: {scores['Hard']:.4f}")
    print(f"Overall: {overall:.4f}")


if __name__ == "__main__":
    main()
