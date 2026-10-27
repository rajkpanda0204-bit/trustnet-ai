
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "fake reviews dataset.csv"
MODEL_PATH = ROOT / "models" / "review_classifier.joblib"
SUMMARY_PATH = ROOT / "reports" / "training_summary.json"
OUTPUT_PATH = ROOT / "reports" / "misclassified_reviews.csv"

RANDOM_STATE = 42


def clean_text(text):
    return " ".join(str(text).lower().split())


def main():
    print("Loading dataset and trained model...")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text_", "label"]).copy()

    df["clean_text"] = df["text_"].map(clean_text)

    # Match the training pipeline's duplicate/conflict handling.
    conflicting = (
        df.groupby("clean_text")["label"].nunique()
    )
    conflicting_texts = conflicting[conflicting > 1].index
    df = df[~df["clean_text"].isin(conflicting_texts)].copy()
    df = df.drop_duplicates(subset="clean_text", keep="first")

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\nTraining summary keys:", list(summary.keys()))
    print(
        "\nImportant: this script must reproduce the exact split "
        "and preprocessing used during training."
    )

    # The training script uses 80% development data and 20% test data.
    development_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Match the second split used in training to create the validation set.
    train_df, validation_df = train_test_split(
        development_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=development_df["label"],
    )

    model = joblib.load(MODEL_PATH)

    # This check helps catch a split mismatch before interpreting results.
    predictions = model.predict(test_df["clean_text"])

    errors = test_df.loc[
        predictions != test_df["label"]
    ].copy()

    errors["predicted_label"] = predictions[
        predictions != test_df["label"]
    ]
    errors["actual_label"] = errors["label"]

    errors["error_type"] = errors.apply(
        lambda row: f"Actual {row['actual_label']} -> "
                    f"Predicted {row['predicted_label']}",
        axis=1,
    )

    errors = errors[
        [
            "category",
            "rating",
            "actual_label",
            "predicted_label",
            "error_type",
            "text_",
        ]
    ]

    errors.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTest rows: {len(test_df):,}")
    print(f"Misclassified rows: {len(errors):,}")
    print("\nErrors by direction:")
    print(errors["error_type"].value_counts().to_string())
    print(f"\nSaved errors to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
