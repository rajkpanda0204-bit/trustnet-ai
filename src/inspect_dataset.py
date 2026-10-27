from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("\n❌ No CSV files found in data/raw/")
        print(f"Expected location: {DATA_DIR}")
        print("Download and extract your review dataset first.")
        return

    print("\n🛡️ TRUSTNET AI — DATASET INSPECTOR")
    print("=" * 45)

    for index, file_path in enumerate(csv_files, start=1):
        print(f"{index}. {file_path.name}")

    # Start with the first CSV; we will verify its columns.
    dataset_path = csv_files[0]

    print(f"\n📂 Loading: {dataset_path.name}")

    try:
        df = pd.read_csv(dataset_path, low_memory=False)
    except Exception as error:
        print(f"❌ Could not read CSV: {error}")
        return

    print("\n✅ Dataset loaded successfully!")
    print("-" * 45)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n📋 COLUMN NAMES")
    for column in df.columns:
        print(f" • {column}")

    print("\n🔎 FIRST FIVE ROWS")
    print(df.head().to_string(index=False, max_cols=12))

    print("\n🧹 MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string())

    print("\n🔁 DUPLICATE ROWS")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\n📊 DATA TYPES")
    print(df.dtypes.to_string())

    # Save a report without changing the original dataset.
    report_path = REPORT_DIR / "dataset_inspection.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("TRUSTNET AI — DATASET INSPECTION\n")
        report.write("=" * 45 + "\n")
        report.write(f"File: {dataset_path.name}\n")
        report.write(f"Rows: {len(df):,}\n")
        report.write(f"Columns: {len(df.columns)}\n\n")
        report.write("COLUMN NAMES\n")
        report.write("\n".join(map(str, df.columns)))
        report.write("\n\nMISSING VALUES\n")
        report.write(df.isnull().sum().to_string())
        report.write("\n\nDATA TYPES\n")
        report.write(df.dtypes.to_string())
        report.write("\n\nFIRST FIVE ROWS\n")
        report.write(df.head().to_string(index=False, max_cols=12))

    print(f"\n📄 Report saved to: {report_path}")
    print("\nNext: identify the review-text and label columns.")


if __name__ == "__main__":
    main()
