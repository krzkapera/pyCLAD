from dataclasses import dataclass
from typing import List, Sequence

import torch


@dataclass
class TaskState:
    """One scorer for a concept: a prompt and the knowledge bank extracted under it."""

    prompt_state: torch.Tensor
    knowledge: torch.Tensor

    def on_cpu(self) -> "TaskState":
        return TaskState(prompt_state=self.prompt_state.cpu(), knowledge=self.knowledge.cpu())


@dataclass
class TaskMemory:
    task_id: int
    key: torch.Tensor
    states: List[TaskState]


class TaskMemoryBank:
    def __init__(self, max_tasks: int = 15):
        self.max_tasks = max_tasks
        self.tasks: List[TaskMemory] = []

    def add_task(self, task_id: int, key: torch.Tensor, states: Sequence[TaskState]):
        if len(self.tasks) >= self.max_tasks:
            raise RuntimeError(f"Memory bank full. Cannot exceed {self.max_tasks} tasks.")
        if not states:
            raise ValueError("A task needs at least one prompt/knowledge state")

        self.tasks.append(
            TaskMemory(
                task_id=task_id,
                key=key.cpu(),
                states=[state.on_cpu() for state in states],
            )
        )

    def task_distances(self, query_features: torch.Tensor) -> torch.Tensor:
        B, Np, C = query_features.shape
        query_flat = query_features.reshape(-1, C)

        distances = []
        for task in self.tasks:
            key = task.key.to(query_features.device)
            min_dist = torch.cdist(query_flat, key).min(dim=1).values
            distances.append(min_dist.reshape(B, Np).sum(dim=1))

        return torch.stack(distances, dim=1)

    def select_tasks(self, query_features: torch.Tensor) -> torch.Tensor:
        return self.task_distances(query_features).argmin(dim=1)

    def get_states(self, task_idx: int) -> List[TaskState]:
        return self.tasks[task_idx].states

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
