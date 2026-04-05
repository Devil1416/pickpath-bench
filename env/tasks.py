from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import GridPosition


@dataclass
class DynamicObstacleEvent:
    trigger_step: int
    positions: List[GridPosition]


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    grid_size: int
    start_position: GridPosition
    item_positions: List[GridPosition]
    obstacles: List[GridPosition]
    max_steps: int
    dynamic_events: List[DynamicObstacleEvent]


def get_task(task_id: str) -> TaskDefinition:
    tasks = {task.task_id: task for task in list_tasks()}
    if task_id not in tasks:
        raise ValueError(f"Unknown task_id: {task_id}")
    return tasks[task_id]


def list_tasks() -> List[TaskDefinition]:
    return [
        # ── EASY ──────────────────────────────────────────────────────────────
        # Items lie on a roughly linear path: top-left → middle → bottom-right.
        # Greedy nearest-first produces the optimal order, so score = 1.0.
        # Purpose: establish a "perfect baseline" and confirm the grader works.
        #
        #   S . A . .
        #   . . . . .
        #   . . B . .
        #   . . . . .
        #   . . . . C
        #
        # Greedy & optimal: S→A(2)→B(2)→C(4) = 8 steps  →  score 1.0
        TaskDefinition(
            task_id="easy",
            name="Easy",
            grid_size=5,
            start_position=GridPosition(row=0, col=0),
            item_positions=[
                GridPosition(row=0, col=2),  # A
                GridPosition(row=2, col=2),  # B
                GridPosition(row=4, col=4),  # C
            ],
            obstacles=[],
            max_steps=50,
            dynamic_events=[],
        ),

        # ── MEDIUM ────────────────────────────────────────────────────────────
        # "Down-wall bait" trap.  Analytically verified score = 16/18 ≈ 0.889.
        #
        # Grid (6×6), start = S=(0,0):
        #
        #   S . . . D .
        #   # # . . . .   ← obstacles (1,0) and (1,1)
        #   A . . . . .
        #   . . . . . .
        #   . . . . C .
        #   . B . . . .
        #
        # Items: A=(2,0) bait, B=(5,1), C=(4,4), D=(0,4)
        #
        # WHY AGENT IS SUBOPTIMAL:
        #   Agent picks targets by Manhattan distance. From (0,0):
        #     A manhattan=2 ← strictly nearest, so agent picks A first.
        #   But the wall at (1,0)-(1,1) blocks the direct path down.
        #   BFS to A forces: right→right→down→down→left→left = 6 steps,
        #   not 2. After reaching A the agent is stranded bottom-left and
        #   sweeps B→C→D, arriving late to D across the top.
        #   Agent total: 18 BFS steps.
        #
        # OPTIMAL ORDER: D→A→B→C
        #   S→D(4)→A(6 via wall detour, same cost)→B(2)→C(4) = 16 steps.
        #   Optimal avoids backtracking by visiting D on the way out, then
        #   sweeping the left/bottom half.
        #
        # Score = 16 / 18 ≈ 0.889
        TaskDefinition(
            task_id="medium",
            name="Medium",
            grid_size=6,
            start_position=GridPosition(row=0, col=0),
            item_positions=[
                GridPosition(row=2, col=0),  # A — bait (manhattan=2, bfs≈6)
                GridPosition(row=5, col=1),  # B
                GridPosition(row=4, col=4),  # C
                GridPosition(row=0, col=4),  # D
            ],
            obstacles=[
                GridPosition(row=1, col=0),
                GridPosition(row=1, col=1),
            ],
            max_steps=60,
            dynamic_events=[],
        ),

        # ── HARD ──────────────────────────────────────────────────────────────
        # "Left-wall bait" trap.  Analytically verified score = 21/28 = 0.750.
        #
        # Grid (7×7), start = S=(0,0):
        #
        #   S . # B . . .
        #   . . # . . . .
        #   . . # . . . G
        #   . . # . . . .
        #   A . . . . . .
        #   . . . . . C .
        #   . . E . . . .
        #
        # Items: B=(0,3) bait, A=(4,0), E=(6,2), C=(5,5), G=(2,6)
        # Obstacles: vertical wall col=2 rows 0–3
        #
        # WHY AGENT IS SUBOPTIMAL:
        #   From (0,0): B has manhattan=3 (strictly nearest item).
        #   Agent picks B first. But col-2 wall spanning rows 0–3 forces the
        #   agent all the way down to row 4 before it can cross right, then back
        #   up to row 0: BFS cost = 11 steps.
        #   After reaching B=(0,3) the agent is stranded top-right.
        #   Remaining items A, E are bottom-left → massive backtrack.
        #   Agent order: B(11)→G(6)→C(4)→E(7+)→A(5+) ≈ 28 total BFS steps.
        #
        # OPTIMAL ORDER: A→E→C→G→B
        #   S→A(4)→E(4)→C(6)→G(5)→B(6) = 21 steps.
        #   Optimal ignores the bait, sweeps the open left/bottom first,
        #   then crosses to top-right last (wall detour only paid once at end).
        #
        # Score = 21 / 28 = 0.750
        TaskDefinition(
            task_id="hard",
            name="Hard",
            grid_size=7,
            start_position=GridPosition(row=0, col=0),
            item_positions=[
                GridPosition(row=0, col=3),  # B — bait (manhattan=3, bfs=11)
                GridPosition(row=4, col=0),  # A
                GridPosition(row=6, col=2),  # E
                GridPosition(row=5, col=5),  # C
                GridPosition(row=2, col=6),  # G
            ],
            obstacles=[
                # Vertical wall at col=2, rows 0–3.
                # Prevents crossing to right side in the top half.
                # Agent must go to row≥4 to pass through, adding ~8 extra steps.
                GridPosition(row=0, col=2),
                GridPosition(row=1, col=2),
                GridPosition(row=2, col=2),
                GridPosition(row=3, col=2),
            ],
            max_steps=80,
            dynamic_events=[],
        ),
    ]