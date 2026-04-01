from __future__ import annotations

import argparse
import json

from src.logtheus_ml.inference import LogAttributeExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    extractor = LogAttributeExtractor(args.model_dir, confidence_threshold=args.threshold)
    result = extractor.predict(args.text)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
