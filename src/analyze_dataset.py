from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

def main():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV file found in data/raw/")
        return

    df = pd.read_csv(csv_files[0])

    print("\n" + "=" * 55)
    print("       TRUSTNET AI — DATASET ANALYSIS")
    print("=" * 55)

    # 1. Label distribution
    print("\n[1] REVIEW LABEL DISTRIBUTION")
    print(df["label"].value_counts(dropna=False).to_string())

    print("\nLabel percentages:")
    percentages = df["label"].value_counts(normalize=True) * 100
    print(percentages.round(2).to_string())

    # 2. Rating distribution
    print("\n[2] STAR RATING DISTRIBUTION")
    print(df["rating"].value_counts().sort_index().to_string())

    # 3. Label versus rating
    print("\n[3] LABEL VS STAR RATING")
    print(pd.crosstab(df["label"], df["rating"]).to_string())

    # 4. Categories
    print("\n[4] REVIEWS PER CATEGORY")
    print(df["category"].value_counts().to_string())

    # 5. Review lengths
    df["review_length"] = df["text_"].astype(str).str.len()

    print("\n[5] REVIEW TEXT LENGTH")
    print(df["review_length"].describe().round(2).to_string())

    print("\nAverage text length by label:")
    print(
        df.groupby("label")["review_length"]
        .mean()
        .round(2)
        .to_string()
    )

    # 6. Duplicate information
    print("\n[6] DUPLICATES")
    print("Exact duplicate rows:", df.duplicated().sum())
    print("Duplicate review texts:", df["text_"].duplicated().sum())

    # 7. Save summary
    output = PROJECT_DIR / "reports" / "dataset_analysis.txt"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        f.write("TRUSTNET AI — DATASET ANALYSIS\n\n")
        f.write("LABEL DISTRIBUTION\n")
        f.write(df["label"].value_counts(dropna=False).to_string())
        f.write("\n\nRATING DISTRIBUTION\n")
        f.write(df["rating"].value_counts().sort_index().to_string())
        f.write("\n\nLABEL VS RATING\n")
        f.write(pd.crosstab(df["label"], df["rating"]).to_string())
        f.write("\n\nCATEGORY DISTRIBUTION\n")
        f.write(df["category"].value_counts().to_string())
        f.write("\n\nREVIEW LENGTH STATISTICS\n")
        f.write(df["review_length"].describe().to_string())

    print(f"\nReport saved to: {output}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()