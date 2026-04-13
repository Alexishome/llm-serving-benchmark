from __future__ import annotations

import csv
import json
from pathlib import Path

from workload.types import WorkloadRequest


LEVAL_SUMMARY_HINTS = ("summ", "report", "review", "patent", "meeting", "tv_show")


class LEvalWorkloadAdapter:
    def __init__(self, token_counter) -> None:
        self._token_counter = token_counter

    def load(self, root_dir: str | Path) -> list[WorkloadRequest]:
        root = Path(root_dir)
        files = sorted((root / "LEval-data").rglob("*.jsonl"))
        requests: list[WorkloadRequest] = []

        for file_path in files:
            task_name = file_path.stem
            task_type = self._infer_task_type(task_name)
            with file_path.open("r", encoding="utf-8") as handle:
                for sample_index, line in enumerate(handle):
                    payload = json.loads(line)
                    document = str(payload.get("input", "")).strip()
                    instructions = payload.get("instructions", [])
                    outputs = payload.get("outputs", [])

                    for instruction_index, instruction in enumerate(instructions):
                        expected_output = outputs[instruction_index] if instruction_index < len(outputs) else ""
                        prompt = self._build_prompt(document=document, instruction=str(instruction))
                        request_id = f"leval-{task_name}-{sample_index:04d}-{instruction_index:02d}"
                        requests.append(
                            WorkloadRequest(
                                request_id=request_id,
                                prompt=prompt,
                                input_tokens=self._token_counter(prompt),
                                max_output_tokens=max(32, self._token_counter(str(expected_output))),
                                task_type=task_type,
                                metadata={
                                    "benchmark": "leval",
                                    "dataset_name": task_name,
                                    "source": payload.get("source", task_name),
                                    "evaluation": payload.get("evaluation"),
                                    "expected_output": expected_output,
                                },
                            )
                        )
        return requests

    @staticmethod
    def _build_prompt(document: str, instruction: str) -> str:
        return (
            "You are answering a task based on the long context below.\n\n"
            f"Document:\n{document}\n\n"
            f"Instruction:\n{instruction}\n"
        )

    @staticmethod
    def _infer_task_type(task_name: str) -> str:
        lowered = task_name.lower()
        if any(hint in lowered for hint in LEVAL_SUMMARY_HINTS):
            return "summarization"
        return "question_answering"


class BLUEWorkloadAdapter:
    def __init__(self, token_counter) -> None:
        self._token_counter = token_counter

    def load(self, root_dir: str | Path) -> list[WorkloadRequest]:
        root = Path(root_dir)
        data_dir = root / "data"
        requests: list[WorkloadRequest] = []
        requests.extend(self._load_biosses(data_dir / "BIOSSES"))
        requests.extend(self._load_relation_tsv(data_dir / "ChemProt", "chemprot", relation_prefix="CPR"))
        requests.extend(self._load_relation_tsv(data_dir / "ddi2013-type", "ddi", relation_prefix="DDI"))
        requests.extend(self._load_hoc(data_dir / "hoc"))
        requests.extend(self._load_bc5cdr_pubtator(data_dir / "BC5CDR"))
        return requests

    def _load_biosses(self, directory: Path) -> list[WorkloadRequest]:
        requests: list[WorkloadRequest] = []
        for split in ("train", "dev", "test"):
            path = directory / f"{split}.tsv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    prompt = (
                        "Assess the semantic similarity of the following biomedical sentences on a scale from 0 to 5.\n\n"
                        f"Sentence 1: {row['sentence1']}\n"
                        f"Sentence 2: {row['sentence2']}\n\n"
                        "Return only the similarity score."
                    )
                    expected_output = str(row["score"])
                    requests.append(
                        WorkloadRequest(
                            request_id=f"blue-biosses-{split}-{row['index']}",
                            prompt=prompt,
                            input_tokens=self._token_counter(prompt),
                            max_output_tokens=max(8, self._token_counter(expected_output)),
                            task_type="question_answering",
                            metadata={
                                "benchmark": "blue",
                                "dataset_name": "biosses",
                                "split": split,
                                "expected_output": expected_output,
                                "task_family": "sentence_similarity",
                            },
                        )
                    )
        return requests

    def _load_relation_tsv(
        self,
        directory: Path,
        dataset_name: str,
        relation_prefix: str,
    ) -> list[WorkloadRequest]:
        requests: list[WorkloadRequest] = []
        for split in ("train", "dev", "test"):
            path = directory / f"{split}.tsv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    prompt = (
                        "Determine the biomedical relation expressed in the sentence below.\n\n"
                        f"Sentence: {row['sentence']}\n\n"
                        "Return the relation label."
                    )
                    expected_output = str(row["label"])
                    requests.append(
                        WorkloadRequest(
                            request_id=f"blue-{dataset_name}-{split}-{row['index']}",
                            prompt=prompt,
                            input_tokens=self._token_counter(prompt),
                            max_output_tokens=max(8, self._token_counter(expected_output)),
                            task_type="information_extraction",
                            metadata={
                                "benchmark": "blue",
                                "dataset_name": dataset_name,
                                "split": split,
                                "expected_output": expected_output,
                                "task_family": "relation_extraction",
                                "relation_prefix": relation_prefix,
                            },
                        )
                    )
        return requests

    def _load_hoc(self, directory: Path) -> list[WorkloadRequest]:
        requests: list[WorkloadRequest] = []
        for split in ("train", "dev", "test"):
            path = directory / f"{split}.tsv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    labels = row.get("labels", "")
                    prompt = (
                        "Assign the relevant Hallmarks of Cancer labels to the biomedical sentence below.\n\n"
                        f"Sentence: {row['sentence']}\n\n"
                        "Return a comma-separated list of labels."
                    )
                    requests.append(
                        WorkloadRequest(
                            request_id=f"blue-hoc-{split}-{row['index']}",
                            prompt=prompt,
                            input_tokens=self._token_counter(prompt),
                            max_output_tokens=max(12, self._token_counter(labels or "none")),
                            task_type="information_extraction",
                            metadata={
                                "benchmark": "blue",
                                "dataset_name": "hoc",
                                "split": split,
                                "expected_output": labels,
                                "task_family": "document_classification",
                            },
                        )
                    )
        return requests

    def _load_bc5cdr_pubtator(self, directory: Path) -> list[WorkloadRequest]:
        requests: list[WorkloadRequest] = []
        split_files = {
            "train": directory / "CDR_TrainingSet.PubTator.txt",
            "dev": directory / "CDR_DevelopmentSet.PubTator.txt",
            "test": directory / "CDR_TestSet.PubTator.txt",
        }
        for split, path in split_files.items():
            if not path.exists():
                continue
            for doc_id, title, abstract, entities in self._iter_pubtator_documents(path):
                prompt = (
                    "Extract all chemical and disease entities from the biomedical document below.\n\n"
                    f"Title: {title}\n\n"
                    f"Abstract: {abstract}\n\n"
                    "Return the entities grouped by type."
                )
                expected_output = json.dumps(entities[:20], ensure_ascii=True)
                requests.append(
                    WorkloadRequest(
                        request_id=f"blue-bc5cdr-{split}-{doc_id}",
                        prompt=prompt,
                        input_tokens=self._token_counter(prompt),
                        max_output_tokens=max(32, self._token_counter(expected_output)),
                        task_type="information_extraction",
                        metadata={
                            "benchmark": "blue",
                            "dataset_name": "bc5cdr",
                            "split": split,
                            "expected_output": expected_output,
                            "task_family": "named_entity_recognition",
                            "entity_count": len(entities),
                        },
                    )
                )
        return requests

    @staticmethod
    def _iter_pubtator_documents(path: Path):
        lines = path.read_text(encoding="utf-8").splitlines()
        block: list[str] = []
        for line in lines + [""]:
            if line.strip():
                block.append(line)
                continue
            if not block:
                continue
            title = ""
            abstract = ""
            entities: list[dict[str, str]] = []
            doc_id = block[0].split("|", 1)[0]
            for entry in block:
                if "|t|" in entry:
                    _, title = entry.split("|t|", 1)
                elif "|a|" in entry:
                    _, abstract = entry.split("|a|", 1)
                else:
                    parts = entry.split("\t")
                    if len(parts) >= 6:
                        entities.append(
                            {
                                "text": parts[3],
                                "type": parts[4],
                                "mesh_id": parts[5],
                            }
                        )
            yield doc_id, title, abstract, entities
            block = []


class ACLWorkloadAdapter:
    def __init__(self, token_counter) -> None:
        self._token_counter = token_counter

    def load_pubhealth(self, root_dir: str | Path, limit: int | None = None) -> list[WorkloadRequest]:
        root = Path(root_dir)
        dataset_dir = root / "PUBHEALTH"
        requests: list[WorkloadRequest] = []
        for split in ("train", "dev", "test"):
            path = dataset_dir / f"{split}.tsv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    prompt = (
                        "Assess the following public-health claim using the evidence article.\n\n"
                        f"Claim: {row['claim']}\n\n"
                        f"Evidence Article:\n{row['main_text']}\n\n"
                        "Return the fact-check label for the claim."
                    )
                    expected_output = str(row["label"])
                    requests.append(
                        WorkloadRequest(
                            request_id=f"acl-pubhealth-{split}-{row['claim_id']}",
                            prompt=prompt,
                            input_tokens=self._token_counter(prompt),
                            max_output_tokens=max(8, self._token_counter(expected_output)),
                            task_type="question_answering",
                            metadata={
                                "benchmark": "acl",
                                "dataset_name": "pubhealth",
                                "split": split,
                                "expected_output": expected_output,
                                "task_family": "fact_checking",
                                "subjects": row.get("subjects"),
                            },
                        )
                    )
                    if limit is not None and len(requests) >= limit:
                        return requests
        return requests

    def load_cochrane(self, root_dir: str | Path, limit: int | None = None) -> list[WorkloadRequest]:
        root = Path(root_dir)
        dataset_dir = root / "cochrane"
        requests: list[WorkloadRequest] = []
        split_files = {
            "train": dataset_dir / "train_v2.json",
            "dev": dataset_dir / "val_v2.json",
            "test_before": dataset_dir / "test_before_cutoff_v2.json",
            "test_after": dataset_dir / "test_after_cutoff_v2.json",
        }
        for split, path in split_files.items():
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    payload = json.loads(line)
                    prompt = (
                        "Summarize the following evidence review abstract into its main clinical conclusion.\n\n"
                        f"Abstract:\n{payload['abstract']}\n"
                    )
                    expected_output = str(payload.get("conclusion", ""))
                    requests.append(
                        WorkloadRequest(
                            request_id=f"acl-cochrane-{split}-{line_index:05d}",
                            prompt=prompt,
                            input_tokens=self._token_counter(prompt),
                            max_output_tokens=max(32, self._token_counter(expected_output)),
                            task_type="summarization",
                            metadata={
                                "benchmark": "acl",
                                "dataset_name": "cochrane",
                                "split": split,
                                "expected_output": expected_output,
                                "task_family": "evidence_summarization",
                                "doi": payload.get("doi"),
                            },
                        )
                    )
                    if limit is not None and len(requests) >= limit:
                        return requests
        return requests

    def load_mimic_bhc(self, root_dir: str | Path, limit: int | None = None) -> list[WorkloadRequest]:
        root = Path(root_dir)
        path = root / "labelled-notes-hospital-course" / "1.2.0" / "mimic-iv-bhc.csv"
        requests: list[WorkloadRequest] = []
        if not path.exists():
            return requests

        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                prompt = (
                    "Summarize the following discharge note into the hospital course.\n\n"
                    f"Clinical Note:\n{row['input']}\n"
                )
                expected_output = str(row["target"])
                requests.append(
                    WorkloadRequest(
                        request_id=f"acl-mimic-bhc-{row['note_id']}",
                        prompt=prompt,
                        input_tokens=int(row.get("input_tokens") or self._token_counter(prompt)),
                        max_output_tokens=max(
                            32,
                            int(row.get("target_tokens") or self._token_counter(expected_output)),
                        ),
                        task_type="summarization",
                        metadata={
                            "benchmark": "acl",
                            "dataset_name": "mimic_bhc",
                            "expected_output": expected_output,
                            "task_family": "clinical_summarization",
                            "note_id": row.get("note_id"),
                        },
                    )
                )
                if limit is not None and len(requests) >= limit:
                    return requests
        return requests
