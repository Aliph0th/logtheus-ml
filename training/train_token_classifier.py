from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", default="models/pretrained", 
                        help="Local directory to cache downloaded models")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(Path(args.train_file))
    val_rows = load_jsonl(Path(args.val_file))

    label2id, id2label = build_label_maps(train_rows + val_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=str(cache_dir))

    train_encoded = [encode_row(row, tokenizer, label2id)
                     for row in train_rows]
    val_encoded = [encode_row(row, tokenizer, label2id) for row in val_rows]

    train_ds = Dataset.from_list(train_encoded)
    val_ds = Dataset.from_list(val_encoded)

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        cache_dir=str(cache_dir),
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            cur_preds = []
            cur_labels = []
            for p, l in zip(prediction, label):
                if l == -100:
                    continue
                cur_preds.append(id2label[int(p)])
                cur_labels.append(id2label[int(l)])
            true_predictions.append(cur_preds)
            true_labels.append(cur_labels)

        if not true_predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        scores = metric.compute(
            predictions=true_predictions, references=true_labels)
        if not scores:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        return {
            "precision": float(scores.get("overall_precision", 0.0)),
            "recall": float(scores.get("overall_recall", 0.0)),
            "f1": float(scores.get("overall_f1", 0.0)),
            "accuracy": float(scores.get("overall_accuracy", 0.0)),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=3e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "model_version": output_dir.name,
        "base_model": args.base_model,
        "metrics": eval_metrics if eval_metrics else {},
        "labels": sorted(label2id.keys()),
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata,
                                                               indent=2), encoding="utf-8")

    print("Training completed. Metrics:")
    print(json.dumps(eval_metrics, indent=2))


def build_label_maps(rows: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    labels = {"O"}
    for row in rows:
        for ent in row.get("entities", []):
            field = ent["label"]
            labels.add(f"B-{field}")
            labels.add(f"I-{field}")

    sorted_labels = sorted(labels)
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_row(row: dict, tokenizer, label2id: dict[str, int]) -> dict:
    text = row["text"]
    entities = row.get("entities", [])

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"]

    labels = [label2id["O"]] * len(offsets)

    for ent in entities:
        start, end, field = ent["start"], ent["end"], ent["label"]
        begin_label = f"B-{field}"
        inside_label = f"I-{field}"
        if begin_label not in label2id or inside_label not in label2id:
            # Skip labels that are outside the configured label space.
            continue

        started = False
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue
            overlaps = not (tok_end <= start or tok_start >= end)
            if not overlaps:
                continue

            if not started:
                labels[i] = label2id[begin_label]
                started = True
            else:
                labels[i] = label2id[inside_label]

    tokenized["labels"] = labels
    return tokenized


if __name__ == "__main__":
    main()
