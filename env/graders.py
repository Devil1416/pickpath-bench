from __future__ import annotations

from collections import deque
from functools import lru_cache

from .models import ActionType, ObservationModel
from .tasks import TaskDefinition, get_task

ACTION_DELTAS: dict[ActionType, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ACTION_ORDER: tuple[ActionType, ...] = ("up", "right", "down", "left")


def _to_tuple(position: object) -> tuple[int, int]:
    row = getattr(position, "row")
    col = getattr(position, "col")
    return row, col


def _active_obstacles(task: TaskDefinition, step_count: int) -> set[tuple[int, int]]:
    obstacles = {_to_tuple(position) for position in task.obstacles}
    for event in task.dynamic_events:
        if step_count >= event.trigger_step:
            obstacles.update(_to_tuple(position) for position in event.positions)
    return obstacles


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


@lru_cache(maxsize=None)
def optimal_steps_for_task(task_id: str) -> int:
    task = get_task(task_id)
    return _optimal_steps(
        task=task,
        start=_to_tuple(task.start_position),
        remaining_items=tuple(sorted(_to_tuple(position) for position in task.item_positions)),
        starting_step_count=0,
    )


def _optimal_steps(
    task: TaskDefinition,
    start: tuple[int, int],
    remaining_items: tuple[tuple[int, int], ...],
    starting_step_count: int,
) -> int:
    item_index = {item: index for index, item in enumerate(remaining_items)}
    target_mask = (1 << len(remaining_items)) - 1

    queue = deque([(start[0], start[1], 0, starting_step_count)])
    visited = {(start[0], start[1], 0, starting_step_count)}

    while queue:
        row, col, collected_mask, step_count = queue.popleft()
        if collected_mask == target_mask:
            return step_count - starting_step_count

        current_obstacles = _active_obstacles(task, step_count)
        for action in ACTION_ORDER:
            delta_row, delta_col = ACTION_DELTAS[action]
            next_row = row + delta_row
            next_col = col + delta_col
            if not (0 <= next_row < task.grid_size and 0 <= next_col < task.grid_size):
                continue
            if (next_row, next_col) in current_obstacles:
                continue

            next_mask = collected_mask
            if (next_row, next_col) in item_index:
                next_mask |= 1 << item_index[(next_row, next_col)]

            next_step_count = step_count + 1
            if next_step_count > task.max_steps:
                continue

            next_state = (next_row, next_col, next_mask, next_step_count)
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    return task.max_steps


def optimal_plan_from_observation(
    task: TaskDefinition, observation: ObservationModel
) -> list[ActionType]:
    remaining_items = tuple(
        sorted((position.row, position.col) for position in observation.item_positions)
    )
    if not remaining_items:
        return []

    item_index = {item: index for index, item in enumerate(remaining_items)}
    target_mask = (1 << len(remaining_items)) - 1
    start = (
        observation.agent_position.row,
        observation.agent_position.col,
        0,
        observation.step_count,
    )

    queue = deque([start])
    parents: dict[tuple[int, int, int, int], tuple[tuple[int, int, int, int], ActionType]] = {}
    visited = {start}
    found_state: tuple[int, int, int, int] | None = None

    while queue:
        row, col, collected_mask, step_count = queue.popleft()
        if collected_mask == target_mask:
            found_state = (row, col, collected_mask, step_count)
            break

        current_obstacles = _active_obstacles(task, step_count)
        for action in ACTION_ORDER:
            delta_row, delta_col = ACTION_DELTAS[action]
            next_row = row + delta_row
            next_col = col + delta_col
            if not (0 <= next_row < task.grid_size and 0 <= next_col < task.grid_size):
                continue
            if (next_row, next_col) in current_obstacles:
                continue

            next_mask = collected_mask
            if (next_row, next_col) in item_index:
                next_mask |= 1 << item_index[(next_row, next_col)]

            next_step_count = step_count + 1
            if next_step_count > task.max_steps:
                continue

            next_state = (next_row, next_col, next_mask, next_step_count)
            if next_state in visited:
                continue

            visited.add(next_state)
            parents[next_state] = ((row, col, collected_mask, step_count), action)
            queue.append(next_state)

    if found_state is None:
        return []

    actions: list[ActionType] = []
    current = found_state
    while current != start:
        previous, action = parents[current]
        actions.append(action)
        current = previous

    actions.reverse()
    return actions


def grade_episode(task_id: str, actual_steps: int, items_collected: int) -> float:
    task = get_task(task_id)
    if actual_steps <= 0 or items_collected <= 0:
        return 0.0

    optimal_steps = optimal_steps_for_task(task_id)
    item_fraction = items_collected / len(task.item_positions)
    step_fraction = optimal_steps / actual_steps
    return clamp_score(step_fraction * item_fraction)

