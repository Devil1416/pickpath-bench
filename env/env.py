from __future__ import annotations

from .models import ActionModel, GridPosition, ObservationModel, RewardModel, StepInfoModel
from .tasks import TaskDefinition, get_task

ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class WarehouseBotEnv:
    def __init__(self, task_id: str = "easy") -> None:
        self._task: TaskDefinition | None = None
        self._agent_position: tuple[int, int] = (0, 0)
        self._remaining_items: set[tuple[int, int]] = set()
        self._picked_items: list[tuple[int, int]] = []
        self._obstacles: set[tuple[int, int]] = set()
        self._step_count = 0
        self._done = False
        self.reset(task_id)

    def reset(self, task_id: str | None = None) -> ObservationModel:
        if task_id is not None:
            self._task = get_task(task_id)
        elif self._task is None:
            self._task = get_task("easy")

        assert self._task is not None
        self._agent_position = self._task.start_position.as_tuple()
        self._remaining_items = {
            position.as_tuple() for position in self._task.item_positions
        }
        self._picked_items = []
        self._obstacles = {position.as_tuple() for position in self._task.obstacles}
        self._step_count = 0
        self._done = False
        return self.state()

    def state(self) -> ObservationModel:
        assert self._task is not None
        return ObservationModel(
            grid_size=self._task.grid_size,
            agent_position=GridPosition(row=self._agent_position[0], col=self._agent_position[1]),
            item_positions=[
                GridPosition(row=row, col=col)
                for row, col in sorted(self._remaining_items)
            ],
            picked_items=[
                GridPosition(row=row, col=col) for row, col in self._picked_items
            ],
            obstacles=[
                GridPosition(row=row, col=col) for row, col in sorted(self._obstacles)
            ],
            step_count=self._step_count,
            task_id=self._task.task_id,
            done=self._done,
            total_items=len(self._task.item_positions),
        )

    def step(self, action: ActionModel | str) -> RewardModel:
        assert self._task is not None
        if isinstance(action, str):
            action = ActionModel(action=action)

        if self._done:
            observation = self.state()
            return RewardModel(
                reward=0.0,
                done=True,
                observation=observation,
                info=StepInfoModel(
                    collected_items=len(self._picked_items),
                    total_items=len(self._task.item_positions),
                    task_complete=not self._remaining_items,
                ),
            )

        reward = -1.0
        invalid_move = False
        item_collected = False
        dynamic_event_triggered = False

        delta_row, delta_col = ACTION_DELTAS[action.action]
        next_row = self._agent_position[0] + delta_row
        next_col = self._agent_position[1] + delta_col

        if not self._is_valid_position(next_row, next_col):
            reward -= 10.0
            invalid_move = True
        else:
            self._agent_position = (next_row, next_col)
            if self._agent_position in self._remaining_items:
                self._remaining_items.remove(self._agent_position)
                self._picked_items.append(self._agent_position)
                reward += 50.0
                item_collected = True

        self._step_count += 1
        dynamic_event_triggered = self._apply_dynamic_obstacles()

        early_finish_bonus = 0.0
        if not self._remaining_items:
            self._done = True
            reward += early_finish_bonus
        elif self._step_count >= self._task.max_steps:
            self._done = True

        observation = self.state()
        return RewardModel(
            reward=reward,
            done=self._done,
            observation=observation,
            info=StepInfoModel(
                invalid_move=invalid_move,
                item_collected=item_collected,
                collected_items=len(self._picked_items),
                total_items=len(self._task.item_positions),
                early_finish_bonus=early_finish_bonus,
                task_complete=self._done and not self._remaining_items,
                dynamic_event_triggered=dynamic_event_triggered,
            ),
        )

    def _is_valid_position(self, row: int, col: int) -> bool:
        assert self._task is not None
        if row < 0 or col < 0 or row >= self._task.grid_size or col >= self._task.grid_size:
            return False
        return (row, col) not in self._obstacles

    def _apply_dynamic_obstacles(self) -> bool:
        assert self._task is not None
        event_triggered = False
        for event in self._task.dynamic_events:
            if event.trigger_step != self._step_count:
                continue

            for position in event.positions:
                self._obstacles.add(position.as_tuple())
            event_triggered = True

        return event_triggered
