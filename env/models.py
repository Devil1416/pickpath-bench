from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ActionType = Literal["up", "down", "left", "right"]


class GridPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    row: int = Field(ge=0)
    col: int = Field(ge=0)

    def as_tuple(self) -> tuple[int, int]:
        return self.row, self.col


class ObservationModel(BaseModel):
    grid_size: int = Field(gt=0)
    agent_position: GridPosition
    item_positions: list[GridPosition] = Field(default_factory=list)
    picked_items: list[GridPosition] = Field(default_factory=list)
    obstacles: list[GridPosition] = Field(default_factory=list)
    step_count: int = Field(ge=0)
    task_id: str
    done: bool = False
    total_items: int = Field(ge=0)
    available_actions: list[ActionType] = Field(
        default_factory=lambda: ["up", "down", "left", "right"]
    )


class ActionModel(BaseModel):
    action: ActionType


class StepInfoModel(BaseModel):
    invalid_move: bool = False
    item_collected: bool = False
    collected_items: int = Field(ge=0)
    total_items: int = Field(ge=0)
    early_finish_bonus: float = 0.0
    task_complete: bool = False
    dynamic_event_triggered: bool = False


class RewardModel(BaseModel):
    reward: float
    done: bool
    observation: ObservationModel
    info: StepInfoModel

