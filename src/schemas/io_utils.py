"""I/O helpers used by the standalone task-generation runner."""

import json
from pathlib import Path
from typing import List

from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.task_schemas import Task


def save_tasks(
    tasks: List[Task], metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save generated tasks to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        "tasks": [task.to_dict() for task in tasks],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
