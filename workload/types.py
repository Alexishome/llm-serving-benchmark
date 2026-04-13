from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkloadRequest:
    request_id: str
    prompt: str
    input_tokens: int
    max_output_tokens: int
    task_type: str
    metadata: dict[str, Any]
