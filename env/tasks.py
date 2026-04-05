from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import GridPosition


class DynamicObstacleEvent(BaseModel):
    trigger_step: int = Field(ge=1)
    positions: list[GridPosition] = Field(default_factory=list)
    description: str


class TaskDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    name: str
    difficulty: str
    description: str
    grid_size: int = Field(ge=5, le=6)
    start_position: GridPosition
    item_positions: list[GridPosition] = Field(default_factory=list)
    obstacles: list[GridPosition] = Field(default_factory=list)
    dynamic_events: list[DynamicObstacleEvent] = Field(default_factory=list)
    max_steps: int = Field(ge=1)


TASKS: dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="easy",
        name="Easy",
        difficulty="easy",
        description=(
            "Collect three items in a compact 5x5 warehouse with no obstacles."
        ),
        grid_size=5,
        start_position=GridPosition(row=0, col=0),
        item_positions=[
            GridPosition(row=0, col=2),
            GridPosition(row=2, col=2),
            GridPosition(row=4, col=4),
        ],
        max_steps=20,
    ),
    "medium": TaskDefinition(
        task_id="medium",
        name="Medium",
        difficulty="medium",
        description=(
            "Collect four items in a 6x6 warehouse with a fixed divider wall and "
            "multiple valid routing choices."
        ),
        grid_size=6,
        start_position=GridPosition(row=0, col=0),
        item_positions=[
            GridPosition(row=0, col=5),
            GridPosition(row=2, col=4),
            GridPosition(row=5, col=0),
            GridPosition(row=5, col=5),
        ],
        obstacles=[
            GridPosition(row=1, col=2),
            GridPosition(row=2, col=2),
            GridPosition(row=3, col=2),
            GridPosition(row=4, col=2),
        ],
        max_steps=28,
    ),
    "hard": TaskDefinition(
        task_id="hard",
        name="Hard",
        difficulty="hard",
        description=(
            "Collect four items in a 6x6 warehouse where a central corridor closes "
            "after step four, forcing a deterministic replan through a lower aisle."
        ),
        grid_size=6,
        start_position=GridPosition(row=0, col=0),
        item_positions=[
            GridPosition(row=0, col=5),
            GridPosition(row=2, col=5),
            GridPosition(row=4, col=5),
            GridPosition(row=5, col=0),
        ],
        obstacles=[
            GridPosition(row=0, col=3),
            GridPosition(row=1, col=3),
            GridPosition(row=3, col=3),
            GridPosition(row=4, col=3),
        ],
        dynamic_events=[
            DynamicObstacleEvent(
                trigger_step=4,
                positions=[GridPosition(row=2, col=3)],
                description="Central aisle closes for safety, blocking the shortest path.",
            )
        ],
        max_steps=36,
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc


def list_tasks() -> list[TaskDefinition]:
    return [TASKS["easy"], TASKS["medium"], TASKS["hard"]]

