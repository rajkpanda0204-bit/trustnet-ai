
from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "review_classifier.joblib"
SUMMARY_PATH = BASE_DIR / "reports" / "training_summary.json"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"

app = Flask(__name__)

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def load_summary():
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


@app.route("/")
def index():
    summary = load_summary()

    return render_template(
        "index.html",
        summary=summary,
    )


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "system": "TRUSTNET AI",
    })


@app.route("/api/scan", methods=["POST"])
def scan_review():
    if model is None:
        return jsonify({
            "error": "Model not found. Train the model first."
        }), 503

    payload = request.get_json(silent=True) or {}
    review = payload.get("review", "")

    if not isinstance(review, str) or not review.strip():
        return jsonify({"error": "Enter a review to analyse."}), 400

    review = review.strip()

    if len(review) > 10000:
        return jsonify({
            "error": "Review is too long. Maximum length is 10,000 characters."
        }), 400

    prediction = str(model.predict([review])[0])

    decision_score = model.decision_function([review])
    score_value = float(decision_score[0])

    # LinearSVC decision score is not a probability.
    classes = [str(value) for value in model.classes_]

    return jsonify({
        "prediction": prediction,
        "decision_score": round(score_value, 4),
        "classes": classes,
        "review_length": len(review),
        "model": "Linear SVM",
        "interpretation": (
            "This is a model prediction, not proof that a review is fake "
            "or genuine. Confirm the dataset label mapping before interpreting "
            "the predicted class."
        ),
    })


@app.route("/api/model-metrics")
def model_metrics():
    summary = load_summary()

    comparison = []
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH).to_dict(
            orient="records"
        )

    return jsonify({
        "best_model": summary.get("best_model", "Not available"),
        "training_rows": summary.get("training_rows"),
        "testing_rows": summary.get("testing_rows"),
        "label_classes": summary.get("label_classes", []),
        "results": comparison,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
