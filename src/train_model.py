
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def build_word_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )


def build_word_character_vectorizer():
    word_features = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
    )

    character_features = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )

    return FeatureUnion([
        ("word", word_features),
        ("character", character_features),
    ])


def build_models():
    return {
        "Logistic Regression — Word TF-IDF": Pipeline([
            ("tfidf", build_word_vectorizer()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "Linear SVM — Word TF-IDF": Pipeline([
            ("tfidf", build_word_vectorizer()),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "Linear SVM — Word + Character TF-IDF": Pipeline([
            ("tfidf", build_word_character_vectorizer()),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def macro_metrics(y_true, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {DATA_DIR}"
        )

    data_path = next(
        (
            path for path in csv_files
            if path.name.lower() == "fake reviews dataset.csv"
        ),
        csv_files[0],
    )

    print(f"\nLoading dataset: {data_path.name}")
    df = pd.read_csv(data_path)

    required_columns = {"text_", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Required columns: {required_columns}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["text_", "label"]].copy()
    df = df.dropna(subset=["text_", "label"])

    df["text_"] = df["text_"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[
        (df["text_"].str.len() > 0)
        & (df["label"].str.len() > 0)
    ].copy()

    # Normalize text to identify exact-text duplicates and conflicts.
    df["text_normalised"] = (
        df["text_"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    label_counts = df.groupby("text_normalised")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index

    print(f"Conflicting texts excluded: {len(conflicting_texts):,}")

    df = df[
        ~df["text_normalised"].isin(conflicting_texts)
    ].drop_duplicates(subset="text_normalised").copy()

    df = df.reset_index(drop=True)

    if df["label"].nunique() < 2:
        raise ValueError("At least two label classes are required.")

    if len(df) < 10:
        raise ValueError("Not enough rows to create dataset splits.")

    print(f"Clean rows: {len(df):,}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    X = df["text_"]
    y = df["label"]

    # Split 1: reserve 20% as an untouched test set.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split 2: reserve 20% of the remaining 80% for validation.
    # Final proportions: 64% train, 16% validation, 20% test.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print("\nDataset split sizes:")
    print(f"Training:   {len(X_train):,} ({len(X_train) / len(X):.1%})")
    print(f"Validation: {len(X_val):,} ({len(X_val) / len(X):.1%})")
    print(f"Test:       {len(X_test):,} ({len(X_test) / len(X):.1%})")

    models = build_models()
    validation_results = []
    trained_candidates = {}

    # Train and compare candidates using validation data only.
    for name, model in models.items():
        print("\n" + "=" * 65)
        print(f"TRAINING CANDIDATE: {name}")
        print("=" * 65)

        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        metrics = macro_metrics(y_val, val_predictions)

        print(f"Validation accuracy:        {metrics['accuracy']:.4f}")
        print(f"Validation macro precision: {metrics['macro_precision']:.4f}")
        print(f"Validation macro recall:    {metrics['macro_recall']:.4f}")
        print(f"Validation macro F1:        {metrics['macro_f1']:.4f}")

        validation_results.append({
            "model": name,
            **metrics,
        })

        trained_candidates[name] = model

    results_df = pd.DataFrame(validation_results).sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Select the candidate using validation macro F1.
    best_name = results_df.iloc[0]["model"]

    print("\n" + "=" * 65)
    print("MODEL SELECTION — VALIDATION RESULTS")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print(f"\nSelected candidate: {best_name}")

    # Refit the selected model on all non-test data.
    # The test set remains untouched until final evaluation.
    final_model = clone(models[best_name])
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    print("\nRefitting selected model on training + validation data...")
    final_model.fit(X_train_val, y_train_val)

    # One final evaluation on the held-out test set.
    print("\nEvaluating selected model on the held-out test set...")
    test_predictions = final_model.predict(X_test)
    test_metrics = macro_metrics(y_test, test_predictions)

    labels = sorted(y.unique())

    final_report = classification_report(
        y_test,
        test_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    final_matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=labels,
    )

    # Save the final model.
    model_path = MODEL_DIR / "review_classifier.joblib"
    joblib.dump(final_model, model_path)

    # Save the final test confusion matrix.
    pd.DataFrame(
        final_matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    # Save the final test classification report.
    with (REPORT_DIR / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(final_report, file, indent=2)

    summary = {
        "best_model": best_name,
        "dataset_rows": int(len(X)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "label_classes": [str(label) for label in labels],
        "selection_metric": "validation_macro_f1",
        "validation_results": validation_results,
        "final_test_metrics": test_metrics,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "note": (
            "Candidates were selected using validation macro F1. "
            "The selected model was refitted on training and validation "
            "data before final evaluation on the held-out test set. "
            "Exact normalized duplicate texts were removed before splitting. "
            "Near-duplicate texts may remain across splits. "
            "CG and OR retain their original dataset meanings; verify "
            "dataset documentation before mapping them to fake or genuine. "
            "Results may not generalize to other review sources."
        ),
    }

    with (REPORT_DIR / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 65)
    print("TRUSTNET AI — FINAL TEST RESULTS")
    print("=" * 65)
    print(f"Selected model:  {best_name}")
    print(f"Test accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Test precision:  {test_metrics['macro_precision']:.4f}")
    print(f"Test recall:     {test_metrics['macro_recall']:.4f}")
    print(f"Test macro F1:   {test_metrics['macro_f1']:.4f}")

    print("\nFinal test classification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=labels,
        zero_division=0,
    ))

    print(f"\nSaved model: {model_path}")
    print(f"Saved validation comparison: {results_path}")
    print(f"Saved test confusion matrix: {REPORT_DIR / 'confusion_matrix.csv'}")
    print(f"Saved test classification report: {REPORT_DIR / 'classification_report.json'}")
    print(f"Saved summary: {REPORT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
