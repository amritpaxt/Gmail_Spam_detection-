"""Safely label new Gmail inbox messages with the local spam classifier.

The watcher reads only the authenticated user's mailbox. It does not delete,
archive, forward, or send any mail. With --apply-labels it adds custom labels:
ML-Spam / ML-Ham and ML-Checked. Running without that flag is a safe dry run.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
LABEL_SPAM = "ML-Spam"
LABEL_HAM = "ML-Ham"
LABEL_CHECKED = "ML-Checked"
LABEL_REVIEW = "ML-Review"


def get_gmail_service(credentials_path: Path, token_path: Path):
    """Authenticate with the browser-based OAuth desktop flow and build Gmail API."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Gmail libraries are missing. Run: python -m pip install -r requirements.txt"
        ) from exc

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"OAuth client file not found: {credentials_path.resolve()}\n"
            "Download the Desktop-app JSON from Google Cloud and save it as secrets/credentials.json."
        )
    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            credentials = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _decode(data: str | None) -> str:
    if not data:
        return ""
    data += "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data.encode("ascii")).decode("utf-8", errors="replace")


def _plain_text_from_payload(payload: dict[str, Any]) -> str:
    mime_type = payload.get("mimeType", "")
    body = _decode(payload.get("body", {}).get("data"))
    if mime_type == "text/plain" and body:
        return body
    text_parts = [_plain_text_from_payload(part) for part in payload.get("parts", [])]
    plain = "\n".join(part for part in text_parts if part)
    if plain:
        return plain
    if mime_type == "text/html" and body:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    return body


def extract_message_content(message: dict[str, Any]) -> tuple[str, str]:
    """Return a subject and readable body from a Gmail message resource."""
    payload = message.get("payload", {})
    headers = {item.get("name", "").lower(): item.get("value", "") for item in payload.get("headers", [])}
    subject = headers.get("subject", "")
    body = _plain_text_from_payload(payload)
    return subject, body


def get_or_create_labels(service) -> dict[str, str]:
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    by_name = {item["name"]: item["id"] for item in existing}
    # Gmail permits only colours from its documented palette.
    for name, colour in ((LABEL_SPAM, {"backgroundColor": "#f6c5be", "textColor": "#000000"}),
                         (LABEL_HAM, {"backgroundColor": "#b9e4d0", "textColor": "#000000"}),
                         (LABEL_REVIEW, {"backgroundColor": "#fef1d1", "textColor": "#000000"}),
                         (LABEL_CHECKED, {"backgroundColor": "#efefef", "textColor": "#000000"})):
        if name not in by_name:
            created = service.users().labels().create(userId="me", body={
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
                "color": colour,
            }).execute()
            by_name[name] = created["id"]
    return {name: by_name[name] for name in (LABEL_SPAM, LABEL_HAM, LABEL_REVIEW, LABEL_CHECKED)}


def remove_project_labels(service, max_messages: int) -> int:
    """Remove only labels owned by this project; messages themselves stay unchanged."""
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    by_name = {item["name"]: item["id"] for item in existing}
    remove_ids = [by_name[name] for name in (LABEL_SPAM, LABEL_HAM, LABEL_REVIEW, LABEL_CHECKED) if name in by_name]
    if not remove_ids:
        print("No ML labels exist yet; nothing to reset.")
        return 0
    removed = 0
    page_token = None
    while True:
        response = service.users().messages().list(
            userId="me", q=f"label:{LABEL_CHECKED}", maxResults=min(max_messages, 500), pageToken=page_token
        ).execute()
        ids = [item["id"] for item in response.get("messages", [])]
        if ids:
            service.users().messages().batchModify(
                userId="me", body={"ids": ids, "removeLabelIds": remove_ids}
            ).execute()
            removed += len(ids)
        page_token = response.get("nextPageToken")
        if not page_token or removed >= max_messages:
            break
    print(f"Removed ML-Spam, ML-Ham, ML-Review, and ML-Checked from {removed} message(s).")
    return removed


def score_message(model: Any, subject: str, body: str) -> tuple[str, float | None]:
    """Classify subject + body; return a score only when the model exposes one."""
    text = f"Subject: {subject}\n\n{body}".strip()
    prediction = int(model.predict([text])[0])
    spam_score: float | None = None
    if hasattr(model, "predict_proba"):
        spam_score = float(model.predict_proba([text])[0][1])
    elif hasattr(model, "decision_function"):
        # LinearSVC's margin is converted to a display score, not a calibrated probability.
        margin = float(model.decision_function([text])[0])
        spam_score = 1 / (1 + math.exp(-max(min(margin, 30), -30)))
    return ("spam" if prediction == 1 else "ham"), spam_score


def apply_conservative_policy(prediction: str, spam_score: float | None, spam_threshold: float) -> str:
    """Apply the user-selected two-label policy: spam at threshold, ham otherwise."""
    if spam_score is None:
        return prediction
    if spam_score >= spam_threshold:
        return "spam"
    return "ham"


def append_event(log_path: Path, event: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def process_inbox(
    service,
    model: Any,
    label_ids: dict[str, str] | None,
    lookback_days: int,
    max_messages: int,
    apply_labels: bool,
    log_path: Path,
    spam_threshold: float,
) -> int:
    """Process unlabelled recent inbox mail exactly once, using ML-Checked as a marker."""
    query = f"in:inbox -label:{LABEL_CHECKED} newer_than:{lookback_days}d"
    listed = service.users().messages().list(userId="me", q=query, maxResults=max_messages).execute()
    messages = listed.get("messages", [])
    for item in messages:
        full = service.users().messages().get(userId="me", id=item["id"], format="full").execute()
        subject, body = extract_message_content(full)
        base_prediction, score = score_message(model, subject, body)
        prediction = apply_conservative_policy(base_prediction, score, spam_threshold)
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "message_id": item["id"],
            "thread_id": full.get("threadId"),
            "prediction": prediction,
            "spam_score": score,
            "labels_applied": apply_labels,
        }
        append_event(log_path, event)
        score_text = "n/a" if score is None else f"{score:.3f}"
        print(f"{prediction.upper():4} score={score_text} id={item['id']}")
        if apply_labels and label_ids:
            label_name = {"spam": LABEL_SPAM, "ham": LABEL_HAM, "review": LABEL_REVIEW}[prediction]
            predicted_label = label_ids[label_name]
            service.users().messages().modify(userId="me", id=item["id"], body={
                "addLabelIds": [predicted_label, label_ids[LABEL_CHECKED]],
            }).execute()
    return len(messages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify recent Gmail inbox mail with the saved local model.")
    parser.add_argument("--credentials", type=Path, default=Path("secrets/credentials.json"))
    parser.add_argument("--token", type=Path, default=Path("secrets/token.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/spam_classifier.joblib"))
    parser.add_argument("--log", type=Path, default=Path("outputs/gmail_predictions.jsonl"))
    parser.add_argument("--interval", type=int, default=60, help="Seconds between checks in watch mode")
    parser.add_argument("--lookback-days", type=int, default=7, help="Only consider inbox messages newer than this")
    parser.add_argument("--max-messages", type=int, default=50, help="Maximum messages per poll")
    parser.add_argument("--spam-threshold", type=float, default=0.80,
                        help="Only scores at or above this receive ML-Spam (default: 0.80)")
    parser.add_argument("--once", action="store_true", help="Process one batch and exit")
    parser.add_argument("--apply-labels", action="store_true", help="Create/apply ML labels; omitted means dry run")
    parser.add_argument("--reset-ml-labels", action="store_true", help="Remove this project's labels from previously processed mail")
    parser.add_argument("--confirm-reset", action="store_true", help="Required together with --reset-ml-labels")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.interval < 10 or args.lookback_days < 1 or args.max_messages < 1:
        print("Interval must be >= 10 seconds; lookback days and max messages must be positive.", file=sys.stderr)
        return 2
    if not 0 < args.spam_threshold < 1:
        print("The spam threshold must be between 0 and 1.", file=sys.stderr)
        return 2
    if args.reset_ml_labels and not args.confirm_reset:
        print("Reset is protected. Re-run with both --reset-ml-labels and --confirm-reset.", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"Saved model not found: {args.model.resolve()}. Run python train.py first.", file=sys.stderr)
        return 2
    try:
        payload = joblib.load(args.model)
        service = get_gmail_service(args.credentials, args.token)
        if args.reset_ml_labels:
            remove_project_labels(service, args.max_messages)
            return 0
        label_ids = get_or_create_labels(service) if args.apply_labels else None
        mode = "LABEL MODE" if args.apply_labels else "DRY RUN (no Gmail changes)"
        print(f"Gmail watcher started: {mode}. Press Ctrl+C to stop.")
        while True:
            processed = process_inbox(
                service, payload["model"], label_ids, args.lookback_days,
                args.max_messages, args.apply_labels, args.log, args.spam_threshold,
            )
            print(f"Processed {processed} message(s).")
            if args.once:
                break
            time.sleep(args.interval)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"Gmail watcher failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nGmail watcher stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
