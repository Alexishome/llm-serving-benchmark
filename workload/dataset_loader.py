from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class DatasetLoader:
    """Load prompt records from JSON or CSV files."""

    SUPPORTED_SUFFIXES = {".json", ".csv"}

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if file_path.suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported dataset format {file_path.suffix}. "
                f"Expected one of: {sorted(self.SUPPORTED_SUFFIXES)}"
            )

        if file_path.suffix == ".json":
            return self._load_json(file_path)
        return self._load_csv(file_path)

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            return [self._normalize_record(record) for record in payload]
        if isinstance(payload, dict) and "records" in payload:
            return [self._normalize_record(record) for record in payload["records"]]
        raise ValueError("JSON dataset must be a list of records or a dict with 'records'.")

    def _load_csv(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [self._normalize_record(row) for row in reader]

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        prompt = record.get("prompt") or record.get("text") or record.get("input")
        if not prompt:
            raise ValueError("Dataset record must include one of: prompt, text, input")

        return {
            "prompt": str(prompt),
            "task_type": str(record.get("task_type", "summarization")),
            "max_tokens": int(record.get("max_tokens", record.get("output_tokens", 128))),
            "metadata": {
                key: value
                for key, value in record.items()
                if key not in {"prompt", "text", "input", "task_type", "max_tokens", "output_tokens"}
            },
        }
