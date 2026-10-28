"""Training and inference utilities for robust spam-message classification."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
SPAM_TOKENS = {"spam", "junk", "phishing", "1", "true", "yes", "positive"}
HAM_TOKENS = {"ham", "not spam", "not_spam", "legitimate", "normal", "0", "false", "no", "negative"}
TEXT_HINTS = ("message", "text", "email", "body", "content", "sms", "mail")
LABEL_HINTS = ("category", "label", "class", "target", "spam", "type", "v1")


@dataclass
class CleaningSummary:
    input_rows: int
    missing_or_blank_messages_removed: int
    invalid_label_rows_removed: int
    duplicate_messages_removed: int
    final_rows: int
    ham_rows: int
    spam_rows: int
    text_column: str
    label_column: str


def _read_data(path: str | Path) -> pd.DataFrame:
    """Read common tabular dataset formats, including files with BOM encodings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(str(exc))
    # Real-world message exports occasionally contain broken quote sequences. Keep
    # the valid, structured rows; downstream validation removes non-label metadata.
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(str(exc))
    raise ValueError("Could not parse the dataset. " + " | ".join(errors))


def _find_column(columns: Iterable[str], hints: tuple[str, ...], kind: str) -> str:
    normalised = {str(column).strip().lower().replace("_", " "): str(column) for column in columns}
    for hint in hints:
        if hint in normalised:
            return normalised[hint]
    for normal, original in normalised.items():
        if any(hint in normal for hint in hints):
            return original
    raise ValueError(
        f"Could not identify a {kind} column. Available columns: {list(columns)}. "
        f"Pass a dataset using a standard {kind} column name."
    )


def _map_labels(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.strip().str.lower()
    mapped = cleaned.map(lambda value: 1 if value in SPAM_TOKENS else (0 if value in HAM_TOKENS else np.nan))
    return mapped


def load_and_clean_dataset(path: str | Path) -> tuple[pd.DataFrame, CleaningSummary]:
    """Load, validate, normalise labels, and remove unusable/duplicate messages."""
    raw = _read_data(path)
    text_column = _find_column(raw.columns, TEXT_HINTS, "message/text")
    label_column = _find_column(raw.columns, LABEL_HINTS, "label")
    if text_column == label_column:
        raise ValueError("Text and label columns resolved to the same column.")

    frame = raw[[label_column, text_column]].copy()
    frame.columns = ["label_raw", "message"]
    frame["message"] = frame["message"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    input_rows = len(frame)
    non_blank = frame["message"].ne("")
    missing_removed = int((~non_blank).sum())
    frame = frame.loc[non_blank].copy()
    frame["label"] = _map_labels(frame["label_raw"])
    valid_label = frame["label"].notna()
    invalid_label_removed = int((~valid_label).sum())
    frame = frame.loc[valid_label].copy()
    if frame["label"].nunique() != 2:
        unknown = sorted(raw[label_column].dropna().astype(str).str.strip().str.lower().unique().tolist())
        raise ValueError(
            "The label column must contain recognised ham and spam classes. Found: " + repr(unknown[:20])
        )
    frame["label"] = frame["label"].astype(int)
    before_dedupe = len(frame)
    # Deduplication before splitting prevents the same message leaking into train and test.
    frame = frame.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    duplicate_removed = before_dedupe - len(frame)
    summary = CleaningSummary(
        input_rows=input_rows,
        missing_or_blank_messages_removed=missing_removed,
        invalid_label_rows_removed=invalid_label_removed,
        duplicate_messages_removed=duplicate_removed,
        final_rows=len(frame),
        ham_rows=int((frame["label"] == 0).sum()),
        spam_rows=int((frame["label"] == 1).sum()),
        text_column=text_column,
        label_column=label_column,
    )
    return frame[["message", "label"]], summary


def clean_message(text: object) -> str:
    """Preserve useful spam signals while standardising URLs, emails and phone numbers."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " emailtoken ", text)
    text = re.sub(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b", " phonetoken ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class MessageStatistics(BaseEstimator, TransformerMixin):
    """Sparse, scale-safe handcrafted features combined with TF-IDF features."""

    def fit(self, X: Iterable[str], y: Any = None) -> "MessageStatistics":
        return self

    def transform(self, X: Iterable[str]) -> csr_matrix:
        rows = []
        for value in X:
            text = str(value)
            tokens = re.findall(r"\b\w+\b", text)
            length = max(len(text), 1)
            words = max(len(tokens), 1)
            rows.append([
                np.log1p(len(text)) / 10,
                np.log1p(len(tokens)) / 10,
                sum(ch.isupper() for ch in text) / length,
                text.count("!") / length,
                sum(ch.isdigit() for ch in text) / length,
                text.count("$") / length,
                np.log1p(max((len(token) for token in tokens), default=0)) / 10,
                sum(len(token) >= 8 for token in tokens) / words,
            ])
        return csr_matrix(np.asarray(rows, dtype=np.float64))


def build_features() -> FeatureUnion:
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            preprocessor=clean_message,
            token_pattern=r"(?u)\b[\w£$]+\b",
            ngram_range=(1, 2), min_df=1, max_df=0.995,
            sublinear_tf=True, max_features=70_000,
        )),
        ("character_tfidf", TfidfVectorizer(
            preprocessor=clean_message, analyzer="char_wb", ngram_range=(3, 5),
            min_df=1, max_features=80_000, sublinear_tf=True,
        )),
        ("message_statistics", MessageStatistics()),
    ])


def build_candidates() -> dict[str, Pipeline]:
    """Candidate classifiers sharing high-signal word, character, and length features."""
    def pipeline(classifier: Any) -> Pipeline:
        return Pipeline([
            ("features", build_features()),
            ("select", SelectKBest(chi2, k=75_000)),
            ("classifier", classifier),
        ])

    return {
        "ComplementNB": pipeline(ComplementNB(alpha=0.08)),
        "LogisticRegression": pipeline(LogisticRegression(C=4.0, max_iter=2_000, solver="liblinear")),
        "LinearSVC": pipeline(LinearSVC(C=1.0, class_weight="balanced")),
    }


def _plot_eda(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = frame.copy()
    eda["class"] = eda["label"].map({0: "ham", 1: "spam"})
    eda["message_length"] = eda["message"].str.len()
    eda["word_count"] = eda["message"].str.findall(r"\b\w+\b").str.len()
    eda["average_word_length"] = eda.apply(
        lambda row: row.message_length / max(row.word_count, 1), axis=1
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    counts = eda["class"].value_counts().reindex(["ham", "spam"], fill_value=0)
    axes[0, 0].bar(counts.index, counts.values, color=["#4C78A8", "#F58518"])
    axes[0, 0].set_title("Class distribution")
    axes[0, 0].set_ylabel("Messages")
    for label, count in zip(counts.index, counts.values):
        axes[0, 0].text(label, count, str(count), ha="center", va="bottom")
    for label, colour in (("ham", "#4C78A8"), ("spam", "#F58518")):
        subset = eda.loc[eda["class"] == label, "message_length"]
        axes[0, 1].hist(subset, bins=45, alpha=.65, label=label, color=colour)
        axes[1, 0].hist(eda.loc[eda["class"] == label, "word_count"], bins=40, alpha=.65, label=label, color=colour)
    axes[0, 1].set_title("Message character length")
    axes[1, 0].set_title("Word-count distribution")
    axes[0, 1].legend(); axes[1, 0].legend()
    box_data = [eda.loc[eda["class"] == label, "average_word_length"] for label in ("ham", "spam")]
    axes[1, 1].boxplot(box_data, tick_labels=["ham", "spam"])
    axes[1, 1].set_title("Average word length")
    axes[1, 1].set_ylabel("Characters per word")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    stop_words = {"the", "a", "an", "to", "and", "is", "in", "of", "for", "on", "i", "you", "u", "my", "me", "it", "your", "this", "that", "with", "at", "be", "are", "have", "or"}
    top_words: dict[str, list[dict[str, Any]]] = {}
    for class_name in ("ham", "spam"):
        all_words = re.findall(r"[a-z]{2,}", " ".join(eda.loc[eda["class"] == class_name, "message"].str.lower()))
        counter = Counter(word for word in all_words if word not in stop_words)
        top_words[class_name] = [{"word": word, "count": count} for word, count in counter.most_common(15)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, class_name, colour in zip(axes, ("ham", "spam"), ("#4C78A8", "#F58518")):
        terms = top_words[class_name][::-1]
        ax.barh([item["word"] for item in terms], [item["count"] for item in terms], color=colour)
        ax.set_title(f"Most frequent {class_name} words")
    fig.tight_layout()
    fig.savefig(output_dir / "top_words.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    insights = {
        "records": int(len(eda)),
        "class_distribution": {key: int(value) for key, value in counts.items()},
        "message_length_by_class": eda.groupby("class")["message_length"].agg(["mean", "median", "min", "max"]).round(2).to_dict(orient="index"),
        "word_count_by_class": eda.groupby("class")["word_count"].agg(["mean", "median", "min", "max"]).round(2).to_dict(orient="index"),
        "average_word_length_by_class": eda.groupby("class")["average_word_length"].mean().round(2).to_dict(),
        "top_words": top_words,
    }
    return insights


def _write_eda_report(summary: CleaningSummary, insights: dict[str, Any], output_dir: Path) -> None:
    length = insights["message_length_by_class"]
    words = insights["word_count_by_class"]
    report = f"""# Mail spam dataset: data-quality and EDA report

## Data loading and cleaning

| Measure | Value |
|---|---:|
| Source text column | `{summary.text_column}` |
| Source label column | `{summary.label_column}` |
| Input rows | {summary.input_rows:,} |
| Blank messages removed | {summary.missing_or_blank_messages_removed:,} |
| Invalid/non-class rows removed | {summary.invalid_label_rows_removed:,} |
| Duplicate messages removed | {summary.duplicate_messages_removed:,} |
| Clean rows | {summary.final_rows:,} |
| Ham | {summary.ham_rows:,} |
| Spam | {summary.spam_rows:,} |

## Message and word-length insights

| Class | Mean message characters | Median characters | Mean words | Median words |
|---|---:|---:|---:|---:|
| Ham | {length['ham']['mean']:.2f} | {length['ham']['median']:.2f} | {words['ham']['mean']:.2f} | {words['ham']['median']:.2f} |
| Spam | {length['spam']['mean']:.2f} | {length['spam']['median']:.2f} | {words['spam']['mean']:.2f} | {words['spam']['median']:.2f} |

`eda_overview.png` visualises class balance, message length, word count, and average word length.
`top_words.png` compares the most frequent non-stopwords in each class.
"""
    (output_dir / "eda_report.md").write_text(report, encoding="utf-8")


def train_and_evaluate(data_path: str | Path, artifact_dir: str | Path = "artifacts", output_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run data cleaning, EDA, 5-fold model selection, hold-out evaluation, and saving."""
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, cleaning_summary = load_and_clean_dataset(data_path)
    insights = _plot_eda(frame, output_dir)
    _write_eda_report(cleaning_summary, insights, output_dir)
    (output_dir / "data_insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")

    X_train, X_test, y_train, y_test = train_test_split(
        frame["message"], frame["label"], test_size=0.20, stratify=frame["label"], random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "spam_f1": make_scorer(f1_score, pos_label=1),
        "spam_precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "spam_recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    }
    scores: list[dict[str, Any]] = []
    candidates = build_candidates()
    for name, pipeline in candidates.items():
        result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        scores.append({"model": name, **{metric: float(np.mean(result[f"test_{metric}"])) for metric in scoring},
                       **{f"{metric}_std": float(np.std(result[f"test_{metric}"])) for metric in scoring}})
    validation = pd.DataFrame(scores).sort_values(["spam_f1", "accuracy"], ascending=False).reset_index(drop=True)
    validation.to_csv(output_dir / "model_validation.csv", index=False)
    best_name = str(validation.loc[0, "model"])
    best_model = candidates[best_name].fit(X_train, y_train)
    prediction = best_model.predict(X_test)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "spam_f1": float(f1_score(y_test, prediction, pos_label=1)),
        "spam_precision": float(precision_score(y_test, prediction, pos_label=1, zero_division=0)),
        "spam_recall": float(recall_score(y_test, prediction, pos_label=1, zero_division=0)),
        "classification_report": classification_report(y_test, prediction, target_names=["ham", "spam"], output_dict=True, zero_division=0),
    }
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, prediction), display_labels=["ham", "spam"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Hold-out test confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Refit on every clean row after the untouched hold-out evaluation has been recorded.
    final_model = build_candidates()[best_name].fit(frame["message"], frame["label"])
    payload = {
        "model": final_model,
        "model_name": best_name,
        "labels": {0: "ham", 1: "spam"},
        "cleaning_summary": asdict(cleaning_summary),
        "validation_metric": "mean 5-fold spam F1; accuracy breaks ties",
        "holdout_metrics": test_metrics,
    }
    joblib.dump(payload, artifact_dir / "spam_classifier.joblib")
    results = {
        "selected_model": best_name,
        "cross_validation": validation.to_dict(orient="records"),
        "holdout_test": test_metrics,
        "saved_model": str((artifact_dir / "spam_classifier.joblib").resolve()),
    }
    (output_dir / "training_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def predict_messages(model_path: str | Path, messages: Iterable[str]) -> list[dict[str, Any]]:
    payload = joblib.load(model_path)
    model = payload["model"]
    items = list(messages)
    labels = model.predict(items)
    output = []
    probabilities = model.predict_proba(items) if hasattr(model, "predict_proba") else None
    for index, (message, label) in enumerate(zip(items, labels)):
        result: dict[str, Any] = {"message": message, "prediction": payload["labels"][int(label)]}
        if probabilities is not None:
            result["spam_probability"] = float(probabilities[index][1])
        output.append(result)
    return output
