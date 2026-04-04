from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .schema import PredictionResult
from ..utils import softmax


class LogAttributeExtractor:
    def __init__(self, model_dir: str, confidence_threshold: float = 0.75, use_onnx: bool | None = None) -> None:
        model_path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_path = model_path
        self.onnx_path = model_path / "model.onnx"
        self.use_onnx = self.onnx_path.exists() if use_onnx is None else use_onnx

        self.model = None
        self.onnx_session = None
        if self.use_onnx:
            self.onnx_session = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"])
            config = json.loads(
                (model_path / "config.json").read_text(encoding="utf-8"))
            self.id2label = {int(k): v for k, v in config["id2label"].items()}
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_path)
            self.model.eval()
            self.id2label = self.model.config.id2label

        self.model_version = self._read_model_version(model_path)
        self.confidence_threshold = confidence_threshold

    def _read_model_version(self, model_path: Path) -> str:
        metadata_file = model_path / "model_metadata.json"
        if metadata_file.exists():
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
            return str(data.get("model_version", "unknown"))
        return "unknown"

    def predict(self, text: str) -> PredictionResult:
        encoded = self.tokenizer(
            text,
            return_tensors="np" if self.use_onnx else "pt",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()

        if self.use_onnx:
            assert self.onnx_session is not None
            onnx_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in encoded:
                onnx_inputs["token_type_ids"] = encoded["token_type_ids"].astype(
                    np.int64)
            outputs = self.onnx_session.run(None, onnx_inputs)
            logits = np.asarray(outputs[0])[0]
        else:
            assert self.model is not None
            with torch.no_grad():
                out = self.model(**encoded)
            logits = out.logits[0].cpu().numpy()

        probs = softmax(logits)
        pred_ids = probs.argmax(axis=-1)
        pred_scores = probs.max(axis=-1)

        grouped_values: dict[str, list[str]] = defaultdict(list)
        grouped_scores: dict[str, list[float]] = defaultdict(list)

        current_label: str | None = None
        current_start: int | None = None
        current_end: int | None = None
        current_scores: list[float] = []

        def flush_current() -> None:
            nonlocal current_label, current_start, current_end, current_scores
            if current_label is None or current_start is None or current_end is None:
                current_label = None
                current_start = None
                current_end = None
                current_scores = []
                return

            value = text[current_start:current_end]
            if value:
                grouped_values[current_label].append(value)
                grouped_scores[current_label].append(float(np.mean(current_scores)) if current_scores else 0.0)

            current_label = None
            current_start = None
            current_end = None
            current_scores = []

        for pred_id, score, (start, end) in zip(pred_ids, pred_scores, offsets):
            if start == end:
                continue

            label = self.id2label[int(pred_id)]
            if label == "O":
                flush_current()
                continue

            prefix, entity = label.split("-", maxsplit=1)

            if prefix == "B" or (current_label is not None and current_label != entity):
                flush_current()
                current_label = entity
                current_start = int(start)
                current_end = int(end)
                current_scores = [float(score)]
            else:
                if current_label is None:
                    current_label = entity
                    current_start = int(start)
                    current_end = int(end)
                    current_scores = [float(score)]
                    continue
                current_label = entity
                current_end = int(end)
                current_scores.append(float(score))

        flush_current()

        attributes: dict[str, str | list[str]] = {}
        low_confidence_attributes: dict[str, str | list[str]] = {}
        attribute_confidence: dict[str, float | list[float]] = {}
        all_scores: list[float] = []

        for label, values in grouped_values.items():
            label_score = float(
                np.mean(grouped_scores[label])) if grouped_scores[label] else 0.0
            all_scores.extend(grouped_scores[label])

            value: str | list[str] = values[0] if len(values) == 1 else values
            confidence_value: float | list[float] = grouped_scores[label][0] if len(grouped_scores[label]) == 1 else grouped_scores[label]
            attribute_confidence[label] = confidence_value

            if label_score >= self.confidence_threshold:
                attributes[label] = value
            else:
                low_confidence_attributes[label] = value

        confidence = float(np.mean(all_scores)) if all_scores else 0.0

        return PredictionResult(
            attributes=attributes,
            low_confidence_attributes=low_confidence_attributes,
            attribute_confidence=attribute_confidence,
            message=text,
            confidence=confidence,
            model_version=self.model_version,
        )
