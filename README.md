# logtheus-ml

ML-only pipeline for log entity extraction and unified attribute naming.

## Goal

Given a raw log line, return a stable contract:

- `attributes`: canonical/unified attributes
- `low_confidence_attributes`: extracted attributes below confidence threshold
- `attribute_confidence`: confidence per extracted attribute
- `message`: original message (or cleaned message in future versions)
- `confidence`: model confidence in [0, 1]
- `model_version`: version of loaded model

## Canonical Schema

The canonical fields are defined in `configs/canonical_fields.json`.

## How Equivalent Fields Are Learned

Model understands equivalent keys through training examples, not hardcoded regex.

Examples in data:

- `lvl=info` labeled as `level`
- `level=info` labeled as `level`
- `err=500` labeled as `status_code` or `error_code` (choose one canonical target)
- `error=500` labeled as the same canonical target

If both variants are mapped to the same label in training data, the model learns them as equivalent.

## Dataset Format

Training input format is JSONL (`data/train.jsonl`, `data/val.jsonl`):

```json
{"id":"1","text":"[auth] failed login for user 123 from 10.1.2.3 code=E401","entities":[{"start":1,"end":5,"label":"service"},{"start":28,"end":32,"label":"user_id"},{"start":38,"end":46,"label":"ip"},{"start":52,"end":56,"label":"error_code"}]}
```

- `start` and `end` are character offsets in `text`.
- `label` must be one of canonical fields.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt
```

## Data Processing

Before training, prepare your raw JSONL files by running:

```bash
python scripts/process_data.py /path/to/raw/logs --data data/train.jsonl --val data/val.jsonl
```

This script:
- **Removes duplicate & similar logs**: Filters logs with `fuzz.ratio >= 94` within each file (requires `rapidfuzz`)
- **Splits into train/val**: Divides data according to the `--p` parameter (train ratio)
- **Renumbers IDs**: Assigns sequential IDs starting from 1

### Options

- `folder`: Path to folder with `.jsonl` files to process
- `--data`, `-d`: Output train file (default: `data/train.jsonl`)
- `--val`, `-v`: Output val file (default: `data/val.jsonl`)
- `--p`: Train ratio in [0, 1], e.g., `0.8` = 80% train, 20% val (default: `0.8`)

## Train

```bash
python training/train_token_classifier.py --train-file data/train.jsonl --val-file data/val.jsonl --output-dir artifacts/model_v1
```

Default base model: `bert-base-uncased` (110M params, good quality/speed balance for logs).

### Model Caching

By default, the base model (e.g., `bert-base-uncased`) is cached locally in `models/pretrained/` directory within the project. This way, you don't need to re-download it for subsequent training runs.

To use a different cache directory:

```bash
python training/train_token_classifier.py --train-file data/train.jsonl --val-file data/val.jsonl --output-dir artifacts/model_v1 --cache-dir /path/to/cache
```

### Alternative Base Models

To use a different model (e.g., DistilBERT for faster training):

```bash
python training/train_token_classifier.py --train-file data/train.jsonl --val-file data/val.jsonl --output-dir artifacts/model_v1 --base-model distilbert-base-uncased
```
## Inference Example

```bash
python scripts/predict.py --model-dir artifacts/model_v1 --text "[auth] failed login for user 123 from 10.1.2.3 code=E401"
```
or interactively:

```bash
python scripts/predict.py --model-dir artifacts/model_v1
```
