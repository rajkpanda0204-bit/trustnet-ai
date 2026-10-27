# TRUSTNET AI
### AI-Powered Fake Review Detection for Trustworthy Recommender Systems

## 1. Overview

TRUSTNET AI is a machine-learning project designed to classify product reviews into two dataset categories: `CG` and `OR`.

The application provides a web-based review scanner that accepts review text, processes it using a trained machine-learning model, and displays the predicted label and decision score.

The system is intended to demonstrate how natural language processing can help identify patterns associated with different types of online reviews.

**Important:** A model prediction is an estimate based on learned patterns. It does not independently prove whether a review is genuine or deceptive.

## 2. Objectives

- Develop a machine-learning model for review classification.
- Apply text preprocessing and TF-IDF feature extraction.
- Compare multiple classification algorithms.
- Evaluate model performance on a held-out test set.
- Provide a web dashboard for interactive review analysis.
- Examine classification errors and model limitations.

## 3. Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF text vectorization
- Linear Support Vector Machine (Linear SVM)
- Flask
- HTML and CSS
- JavaScript
- Joblib

## 4. Dataset

The project uses `fake reviews dataset.csv`.

The dataset contains the following columns:

| Column | Description |
|---|---|
| `category` | Product category |
| `rating` | Review rating |
| `label` | Dataset classification label |
| `text_` | Review text |

The dataset contains approximately 40,432 original rows. After cleaning, conflicting-label checks, and duplicate removal, 40,405 rows remained in the training pipeline.

The exact meanings of `CG` and `OR` should be verified against the dataset's authoritative documentation before interpreting the classes as fake and genuine reviews.

## 5. Methodology

### Data preprocessing

The training pipeline:
- Removes rows with missing review text or labels.
- Normalizes review text to lowercase and standardizes whitespace.
- Excludes normalized texts associated with conflicting labels.
- Removes duplicate normalized reviews.
- Splits the dataset into training, validation, and test sets.

### Dataset split

| Split | Samples | Percentage |
|---|---:|---:|
| Training | 25,859 | 64% |
| Validation | 6,465 | 16% |
| Test | 8,081 | 20% |
| Total | 40,405 | 100% |

### Feature extraction

TF-IDF converts review text into numerical features.

The selected model combines:
- Word-level TF-IDF features using unigrams and bigrams.
- Character-level TF-IDF features using character sequences.

Combining word and character features allows the classifier to learn patterns at different levels of the text.

### Model selection

Three candidate models were compared:

1. Logistic Regression with word-level TF-IDF.
2. Linear SVM with word-level TF-IDF.
3. Linear SVM with combined word- and character-level TF-IDF.

The winning candidate was selected using validation macro F1. It was then refitted on the combined training and validation data and evaluated on the held-out test set.

## 6. Results

The selected model was **Linear SVM with word- and character-level TF-IDF**.

| Metric | Result |
|---|---:|
| Validation accuracy | 95.34% |
| Test accuracy | 95.21% |
| Test macro F1 | 95.21% |
| Test samples | 8,081 |
| Incorrect test predictions | 387 |

### Confusion matrix

| Actual label | Predicted CG | Predicted OR |
|---|---:|---:|
| CG | 3,836 | 202 |
| OR | 185 | 3,858 |

These metrics describe performance on the supplied dataset and its held-out test split. They do not guarantee equivalent performance on reviews from other platforms or distributions.

## 7. Web Application

The Flask application provides:
- An interactive review scanner.
- Predicted dataset label.
- Linear SVM decision score.
- Review-length information.
- Model performance information.
- API endpoints for health checks, review scanning, and model metrics.

The decision score is not a probability or a calibrated confidence percentage.

## 8. Project Structure

```text
trustnet-ai/
├── app.py
├── data/
│   └── raw/
│       └── fake reviews dataset.csv
├── models/
│   └── review_classifier.joblib
├── reports/
│   ├── classification_report.json
│   ├── confusion_matrix.csv
│   ├── dataset_analysis.txt
│   ├── dataset_inspection.txt
│   ├── misclassified_reviews.csv
│   ├── model_comparison.csv
│   └── training_summary.json
├── src/
│   ├── analyze_dataset.py
│   ├── analyze_errors.py
│   ├── inspect_dataset.py
│   ├── predict_review.py
│   └── train_model.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Prerequisites

- Python 3.11 or a compatible Python version.
- pip.

### Setup

Create and activate a virtual environment if one does not already exist.

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start the application

From the project root:

```bash
python app.py
```

Open the local address printed by Flask in your browser. The default development address is commonly `http://127.0.0.1:5000/`.

### Retrain the model

To repeat model training and evaluation:

```bash
python src/train_model.py
```

Retraining replaces the saved model and regenerates the training reports.

## 10. Limitations

- Predictions depend on patterns learned from the training dataset.
- Dataset labels may not perfectly represent real-world review authenticity.
- Near-duplicate checks on a small sample do not rule out all train/test leakage.
- Writing style alone cannot conclusively establish review authorship.
- The decision score is not a calibrated probability.
- External validation is needed before applying the system to other review platforms.

## 11. Future Improvements

- Evaluate the model on an independent dataset.
- Investigate near-duplicate leakage across data splits.
- Add calibrated probabilities if appropriate.
- Improve the user interface and prediction explanations.
- Monitor false-positive and false-negative rates.
- Evaluate robustness against newer AI-generated reviews.

## 12. Conclusion

TRUSTNET AI demonstrates an end-to-end machine-learning workflow for review-text classification, including data preprocessing, model comparison, held-out evaluation, error analysis, and web-based inference.

The selected Linear SVM achieved 95.21% accuracy on the held-out test set. Further independent evaluation is necessary to establish how well it generalizes to reviews outside the training dataset.