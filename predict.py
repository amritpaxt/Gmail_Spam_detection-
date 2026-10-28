"""Predict spam/ham labels for one or more new messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.spam_pipeline import predict_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify new mail/SMS messages.")
    parser.add_argument("messages", nargs="*", help="Messages to classify")
    parser.add_argument("--file", type=Path, help="Text file containing one message per line")
    parser.add_argument("--model", default="artifacts/spam_classifier.joblib", help="Saved model path")
    args = parser.parse_args()
    messages = args.messages
    if args.file:
        messages.extend(line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not messages:
        parser.error("Provide a message argument or --file messages.txt")
    print(json.dumps(predict_messages(args.model, messages), indent=2))


if __name__ == "__main__":
    main()
