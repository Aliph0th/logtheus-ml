from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logtheus_ml.inference import LogAttributeExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=False)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    extractor = LogAttributeExtractor(args.model_dir, confidence_threshold=args.threshold)

    if args.text:
        result = extractor.predict(args.text)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=True))
        return

    print("Model loaded. Enter log lines (Ctrl+C or Ctrl+D to stop).")
    while True:
        try:
            text = input("log> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nStopped.")
            break

        if not text:
            continue

        result = extractor.predict(text)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
