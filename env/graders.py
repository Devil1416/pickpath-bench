from __future__ import annotations

from collections import deque
from itertools import permutations
from typing import Set, Tuple

from .tasks import get_task


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

    # 🔥 TRY ALL ORDERINGS (this makes grader smarter than agent)
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

    # small epsilon to avoid 0 and 1
    eps = 1e-6

    if actual_steps <= 0:
        return eps

    optimal_steps = optimal_steps_for_task(task_id)

    efficiency = optimal_steps / actual_steps
    completion = items_collected / total_items

    score = efficiency * completion

    # clamp STRICTLY between (0,1)
    score = max(eps, min(1.0 - eps, score))

    return score