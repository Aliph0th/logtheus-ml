import os
import json
import argparse
from typing import List, Tuple
from rapidfuzz import fuzz


def pick_split_indices(n: int, val_ratio: float = 0.2) -> Tuple[set, set]:
    if n <= 0:
        return set(), set()
    n_val = int(round(n * val_ratio))
    if n_val <= 0:
        return set(range(n)), set()
    if n_val >= n:
        return set(), set(range(n))
    step = n / n_val
    val_indices = set()
    for i in range(n_val):
        idx = int(round(i * step + step / 2.0))
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        while idx in val_indices and idx + 1 < n:
            idx += 1
        if idx in val_indices:
            idx2 = idx - 1
            while idx2 in val_indices and idx2 >= 0:
                idx2 -= 1
            if idx2 >= 0:
                idx = idx2
        val_indices.add(idx)
    train_indices = set(range(n)) - val_indices
    return train_indices, val_indices


def remove_similar_items(items: List[dict], similarity_threshold: float = 94) -> List[dict]:
    """Remove items with duplicate or very similar text fields.
    
    Args:
        items: List of JSON objects with 'text' field
        similarity_threshold: fuzz.ratio threshold (0-100) to consider items as similar
        
    Returns:
        Filtered list with similar items removed
    """
    if not items:
        return []
    
    filtered = []
    for item in items:
        if not isinstance(item, dict) or "text" not in item:
            filtered.append(item)
            continue
        
        current_text = item.get("text", "")
        is_similar = False
        
        # Compare with already filtered items
        for existing_item in filtered:
            if not isinstance(existing_item, dict) or "text" not in existing_item:
                continue
            existing_text = existing_item.get("text", "")
            if fuzz.ratio(current_text, existing_text) >= similarity_threshold:
                is_similar = True
                break
        
        if not is_similar:
            filtered.append(item)
    
    return filtered


def process_folder(folder: str, output_data: str, output_val: str, val_ratio: float = 0.2):
    data_items = []
    val_items = []
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(".jsonl")
    )
    if not files:
        print("No .jsonl files found in the folder.")
        return
    for fname in files:
        path = os.path.join(folder, fname)
        print(f"Processing file: {path}")
        lines = []
        with open(path, "r", encoding="utf-8") as fin:
            for line_no, raw in enumerate(fin, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON error in file {path}, line {line_no}: {e}\nLine text: {raw!r}")
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected JSON object (dict) in file {path}, line {line_no}, got {type(obj)}")
                lines.append(obj)
        
        # Remove similar items from this file before splitting
        original_count = len(lines)
        lines = remove_similar_items(lines, similarity_threshold=94)
        removed_count = original_count - len(lines)
        if removed_count > 0:
            print(f"  Removed {removed_count} similar items (kept {len(lines)})")
        
        n = len(lines)
        if n == 0:
            print(
                f"  Warning: file {path} is empty or has no valid JSON objects, skipping.")
            continue
        train_idx, val_idx = pick_split_indices(n, val_ratio=val_ratio)
        for i in range(n):
            if i in val_idx:
                val_items.append(lines[i])
            else:
                data_items.append(lines[i])
        print(
            f"  Taken {len(train_idx)} items for train, {len(val_idx)} items for val from {path}")
    print(f"Writing train file: {output_data} (total {len(data_items)} items)")
    with open(output_data, "w", encoding="utf-8") as fout_data:
        counter = 1
        for obj in data_items:
            obj["id"] = str(counter)
            fout_data.write(json.dumps(obj, ensure_ascii=False) + "\n")
            counter += 1
    print(f"Writing val file: {output_val} (total {len(val_items)} items)")
    with open(output_val, "w", encoding="utf-8") as fout_val:
        counter = 1
        for obj in val_items:
            obj["id"] = str(counter)
            fout_val.write(json.dumps(obj, ensure_ascii=False) + "\n")
            counter += 1
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split JSONL files in a folder into data/val, remove similar items, and renumber ids starting from 1.")
    parser.add_argument(
        "folder", help="Path to folder containing .jsonl files")
    parser.add_argument("--data", "-d", default="data/train.jsonl",
                        help="Output train file (default: data/train.jsonl)")
    parser.add_argument("--val", "-v", default="data/val.jsonl",
                        help="Output val file (default: data/val.jsonl)")
    parser.add_argument("--p", "-p", type=float, default=0.8,
                        help="Train ratio (0-1), e.g., 0.8 = 80%% train, 20%% val (default: 0.8)")
    args = parser.parse_args()
    
    # Validate p parameter
    if not 0 < args.p < 1:
        raise ValueError("--p must be between 0 and 1 (exclusive)")
    
    val_ratio = 1 - args.p
    process_folder(args.folder, args.data, args.val, val_ratio=val_ratio)
