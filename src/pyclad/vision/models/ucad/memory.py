import torch
from dataclasses import dataclass
from typing import List


@dataclass
class TaskMemory:
    task_id: int
    key: torch.Tensor
    prompt_state: torch.Tensor
    knowledge: torch.Tensor


class TaskMemoryBank:
    def __init__(self, max_tasks: int = 15):
        self.max_tasks = max_tasks
        self.tasks: List[TaskMemory] = []

    def add_task(
        self,
        task_id: int,
        key: torch.Tensor,
        prompt_state: torch.Tensor,
        knowledge: torch.Tensor,
    ):
        if len(self.tasks) >= self.max_tasks:
            raise RuntimeError(f"Memory bank full. Cannot exceed {self.max_tasks} tasks.")

        memory = TaskMemory(
            task_id=task_id,
            key=key.cpu(),
            prompt_state=prompt_state.cpu(),
            knowledge=knowledge.cpu(),
        )
        self.tasks.append(memory)

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

    def get_prompt_state(self, task_idx: int) -> torch.Tensor:
        return self.tasks[task_idx].prompt_state

    def get_knowledge(self, task_idx: int) -> torch.Tensor:
        return self.tasks[task_idx].knowledge

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
