import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaskMemory:
    """Stores the necessary components for a single task/concept."""
    task_id: int
    key: torch.Tensor          # Shape: (key_size, C)
    prompt_state: torch.Tensor # Shape: (num_layers, 2, prompt_length, num_heads, head_dim)
    knowledge: torch.Tensor    # Shape: (knowledge_size, C)


class TaskMemoryBank:
    """
    Manages the Continual Prompting Module's memory.
    Stores keys (for task selection), prompts (for extraction), and knowledge (for scoring).
    """
    def __init__(self, max_tasks: int = 15):
        self.max_tasks = max_tasks
        self.tasks: List[TaskMemory] = []
        
    def add_task(
        self, 
        task_id: int, 
        key: torch.Tensor, 
        prompt_state: torch.Tensor, 
        knowledge: torch.Tensor
    ):
        """Adds a new task to the memory bank."""
        if len(self.tasks) >= self.max_tasks:
            raise RuntimeError(f"Memory bank full. Cannot exceed {self.max_tasks} tasks.")
            
        memory = TaskMemory(
            task_id=task_id,
            key=key.cpu(),
            prompt_state=prompt_state.cpu(),
            knowledge=knowledge.cpu()
        )
        self.tasks.append(memory)

    def select_task(self, query_features: torch.Tensor) -> int:
        """
        Selects the best matching task for the given query features.
        Matches Equation 4 in the UCAD paper.
        
        Args:
            query_features: Tensor of shape (B, Np, C) containing features extracted
                            from the frozen backbone.
                            
        Returns:
            The task_id of the best matching task.
        """
        if not self.tasks:
            raise RuntimeError("Memory bank is empty. Cannot select a task.")
            
        if len(self.tasks) == 1:
            return 0
            
        B, Np, C = query_features.shape
        
        # Flatten query features to (B*Np, C)
        query_flat = query_features.reshape(-1, C)
        
        best_task_idx = 0
        min_total_distance = float('inf')
        
        # In a real implementation for large batches, we would do this 
        # on a per-image basis. For simplicity (and as often done in CAD),
        # we assume the batch belongs to a single task and aggregate distances.
        # If we need per-image task selection, this logic should return a list of task_ids.
        
        for idx, task in enumerate(self.tasks):
            # key shape: (key_size, C)
            key = task.key.to(query_features.device)
            
            # Compute distance from each query feature to the nearest key feature
            distances = torch.cdist(query_flat, key)  # (B*Np, key_size)
            
            # Min distance to key for each query feature
            min_dist, _ = torch.min(distances, dim=1)  # (B*Np,)
            
            # Aggregate distance for the entire batch
            total_dist = torch.sum(min_dist).item()
            
            if total_dist < min_total_distance:
                min_total_distance = total_dist
                best_task_idx = idx
                
        return best_task_idx

    def get_prompt_state(self, task_idx: int) -> torch.Tensor:
        """Retrieves the prompt state for a given task index."""
        if task_idx >= len(self.tasks):
            raise IndexError(f"Task index {task_idx} out of range (max {len(self.tasks)-1})")
        return self.tasks[task_idx].prompt_state

    def get_knowledge(self, task_idx: int) -> torch.Tensor:
        """Retrieves the knowledge bank for a given task index."""
        if task_idx >= len(self.tasks):
            raise IndexError(f"Task index {task_idx} out of range (max {len(self.tasks)-1})")
        return self.tasks[task_idx].knowledge

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
