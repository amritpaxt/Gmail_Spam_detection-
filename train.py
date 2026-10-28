"""Command-line entry point for the complete mail-spam workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.spam_pipeline import train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate a mail spam detector.")
    parser.add_argument("--data", default="data/email.csv", help="CSV/TSV/XLSX/JSON dataset path")
    parser.add_argument("--artifacts", default="artifacts", help="Directory for the saved model")
    parser.add_argument("--outputs", default="outputs", help="Directory for EDA and evaluation outputs")
    args = parser.parse_args()
    try:
        results = train_and_evaluate(args.data, args.artifacts, args.outputs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2
    print(f"Selected model: {results['selected_model']}")
    print(f"Hold-out accuracy: {results['holdout_test']['accuracy']:.4f}")
    print(f"Hold-out spam F1: {results['holdout_test']['spam_f1']:.4f}")
    print(f"Saved model: {results['saved_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
