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
            input_ids = encoded["input_ids"][0].tolist()
        else:
            assert self.model is not None
            with torch.no_grad():
                out = self.model(**encoded)
            logits = out.logits[0].cpu().numpy()
            input_ids = encoded["input_ids"][0].tolist()

        probs = softmax(logits)
        pred_ids = probs.argmax(axis=-1)
        pred_scores = probs.max(axis=-1)

        grouped_values: dict[str, list[str]] = defaultdict(list)
        grouped_scores: dict[str, list[float]] = defaultdict(list)

        current_label = None
        current_value = []
        current_scores = []

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        for token, pred_id, score, (start, end) in zip(tokens, pred_ids, pred_scores, offsets):
            if start == end:
                continue

            label = self.id2label[int(pred_id)]
            if label == "O":
                if current_label and current_value:
                    grouped_values[current_label].append(
                        "".join(current_value).replace("##", ""))
                    grouped_scores[current_label].append(
                        float(np.mean(current_scores)))
                current_label = None
                current_value = []
                current_scores = []
                continue

            prefix, entity = label.split("-", maxsplit=1)

            normalized_piece = token.replace("##", "")
            if prefix == "B" or (current_label and current_label != entity):
                if current_label and current_value:
                    grouped_values[current_label].append(
                        "".join(current_value))
                    grouped_scores[current_label].append(
                        float(np.mean(current_scores)))
                current_label = entity
                current_value = [normalized_piece]
                current_scores = [float(score)]
            else:
                current_label = entity
                current_value.append(normalized_piece)
                current_scores.append(float(score))

        if current_label and current_value:
            grouped_values[current_label].append("".join(current_value))
            grouped_scores[current_label].append(
                float(np.mean(current_scores)))

        attributes: dict[str, str | list[str]] = {}
        unknown_attributes: dict[str, str | list[str]] = {}
        all_scores: list[float] = []

        for label, values in grouped_values.items():
            label_score = float(
                np.mean(grouped_scores[label])) if grouped_scores[label] else 0.0
            all_scores.extend(grouped_scores[label])
            value: str | list[str] = values[0] if len(values) == 1 else values
            if label_score >= self.confidence_threshold:
                attributes[label] = value
            else:
                unknown_attributes[label] = value

        confidence = float(np.mean(all_scores)) if all_scores else 0.0

        return PredictionResult(
            attributes=attributes,
            unknown_attributes=unknown_attributes,
            message=text,
            confidence=confidence,
            model_version=self.model_version,
        )
