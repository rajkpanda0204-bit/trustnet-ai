from pathlib import Path
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "review_classifier.joblib"


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src/train_model.py")
        return

    model = joblib.load(MODEL_PATH)

    print("\n" + "=" * 55)
    print("          TRUSTNET AI | REVIEW SCANNER")
    print("=" * 55)

    print("\nEnter a product review below.")
    print("Type 'exit' to quit.\n")

    while True:
        review = input("Review > ").strip()

        if review.lower() == "exit":
            print("\nInvestigation session ended.")
            break

        if not review:
            print("Please enter a review.\n")
            continue

        # The current model uses review text only.
        prediction = model.predict([review])[0]

        print("\n" + "-" * 45)
        print("ANALYSIS RESULT")
        print("-" * 45)
        print(f"Predicted dataset label: {prediction}")

        # LinearSVC's decision score is not a probability.
        if hasattr(model, "decision_function"):
            score = model.decision_function([review])

            if len(score.shape) == 1:
                print(f"Decision score: {score[0]:.4f}")
                print(
                    "Note: This is a model decision score, "
                    "not a confidence percentage."
                )

        print(
            "\nInterpretation: Check the dataset documentation "
            "to establish what this label represents."
        )
        print("-" * 45 + "\n")


if __name__ == "__main__":
    main()