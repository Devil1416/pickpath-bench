from __future__ import annotations

from collections import deque
from itertools import permutations
from typing import Set, Tuple

from .tasks import get_task

MIN_SCORE = 0.01
MAX_SCORE = 0.99


def clamp_score(score: float) -> float:
    """Keep scores safely away from endpoints even after downstream rounding."""
    return max(MIN_SCORE, min(MAX_SCORE, score))


def _bfs_shortest_path(
    start: Tuple[int, int],
    target: Tuple[int, int],
    grid_size: int,
    obstacles: Set[Tuple[int, int]],
) -> int:
    if start == target:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (r, c), dist = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue

            if (nr, nc) in obstacles:
                continue

            if (nr, nc) in visited:
                continue

            if (nr, nc) == target:
                return dist + 1

            visited.add((nr, nc))
            queue.append(((nr, nc), dist + 1))

    return float("inf")


def optimal_steps_for_task(task_id: str) -> int:
    task = get_task(task_id)

    obstacles = {(o.row, o.col) for o in task.obstacles}

    start = (task.start_position.row, task.start_position.col)
    items = [(item.row, item.col) for item in task.item_positions]

    best = float("inf")

    for order in permutations(items):
        current = start
        total = 0

        for item in order:
            dist = _bfs_shortest_path(current, item, task.grid_size, obstacles)
            total += dist
            current = item

        best = min(best, total)

    return best


def grade_episode(
    task_id: str,
    actual_steps: int,
    items_collected: int,
) -> float:
    task = get_task(task_id)

    total_items = len(task.item_positions)

    if actual_steps <= 0:
        return MIN_SCORE

    if total_items == 0:
        return MIN_SCORE

    optimal_steps = optimal_steps_for_task(task_id)

    if optimal_steps == float("inf"):
        return MIN_SCORE

    efficiency = optimal_steps / actual_steps
    completion = items_collected / total_items

    score = efficiency * completion

    # Always clamp — score == 1.0 (perfect run on Easy) was failing validation
    return clamp_score(score)
