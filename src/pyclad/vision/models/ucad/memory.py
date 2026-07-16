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

    def select_task(self, query_features: torch.Tensor) -> int:
        if not self.tasks:
            raise RuntimeError("Memory bank is empty. Cannot select a task.")

        if len(self.tasks) == 1:
            return 0

        B, Np, C = query_features.shape
        query_flat = query_features.reshape(-1, C)

        best_task_idx = 0
        min_total_distance = float("inf")

        for idx, task in enumerate(self.tasks):
            key = task.key.to(query_features.device)
            distances = torch.cdist(query_flat, key)
            min_dist, _ = torch.min(distances, dim=1)
            total_dist = torch.sum(min_dist).item()

            if total_dist < min_total_distance:
                min_total_distance = total_dist
                best_task_idx = idx

        return best_task_idx

    def get_prompt_state(self, task_idx: int) -> torch.Tensor:
        if task_idx >= len(self.tasks):
            raise IndexError(f"Task index {task_idx} out of range (max {len(self.tasks)-1})")

        return self.tasks[task_idx].prompt_state

    def get_knowledge(self, task_idx: int) -> torch.Tensor:
        if task_idx >= len(self.tasks):
            raise IndexError(f"Task index {task_idx} out of range (max {len(self.tasks)-1})")

        return self.tasks[task_idx].knowledge

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
