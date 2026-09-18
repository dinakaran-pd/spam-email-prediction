"""
train.py
--------
Job of this file: put everything together to actually TRAIN the model.

Steps:
1. Load the dataset (dataset.py)
2. Split into train/test sets
3. Convert text to TF-IDF vectors (preprocessor.py)
4. Train a Naive Bayes classifier (a simple, fast, and surprisingly
   strong algorithm for text classification / spam detection)
5. Evaluate accuracy on the test set
6. Save the trained model AND the preprocessor to disk (.pkl files)
   so app.py / main.py can load them later without retraining.

Run this file with:  python train.py
"""

import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

from Dataset import load_dataset
from Preprocessor import TextPreprocessor


def train_model(csv_path="spam.csv"):
    # 1. Load data
    print("Loading dataset...")
    df = load_dataset(csv_path)
    print(f"Loaded {len(df)} rows ({df['label'].sum()} spam, "
          f"{(df['label'] == 0).sum()} not spam)")

    X = df["text"]
    y = df["label"]

    # 2. Split into training (80%) and testing (20%) sets
    # stratify=y keeps the same spam/ham ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Convert text -> TF-IDF numeric vectors
    print("Vectorizing text...")
    preprocessor = TextPreprocessor(max_features=3000)
    X_train_vec = preprocessor.fit_transform(X_train)
    X_test_vec = preprocessor.transform(X_test)

    # 4. Train the model
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 5. Evaluate on the held-out test set
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest set accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    # 5b. Cross-validation: a more trustworthy set of metrics.
    # A single train/test split can accidentally land on an "easy" split
    # and report a suspicious, inflated 100% on precision/recall/F1 even
    # when accuracy alone looks more realistic — that mismatch is exactly
    # what makes a dashboard look untrustworthy. To avoid that, we compute
    # ALL FOUR headline metrics (accuracy, precision, recall, F1) the same
    # way: 5-fold cross-validation, training and testing the model 5
    # separate times on different slices of data and averaging the
    # results. This is the standard, credible way to report model
    # performance, and keeps every number on the dashboard consistent.
    if len(df) < 200:
        print(
            f"\nNote: your dataset only has {len(df)} rows. Metrics from "
            f"such a small dataset can look unrealistically perfect (or "
            f"swing a lot between runs). For a more convincing, stable "
            f"result, aim for at least a few hundred rows if you can."
        )

    print("\nRunning 5-fold cross-validation for reliable metrics...")
    X_all_vec = preprocessor.transform(X)  # reuse vocabulary learned on X_train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        MultinomialNB(),
        X_all_vec,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
    )
    cv_accuracy_mean = cv_results["test_accuracy"].mean()
    cv_accuracy_std = cv_results["test_accuracy"].std()
    cv_precision_mean = cv_results["test_precision"].mean()
    cv_precision_std = cv_results["test_precision"].std()
    cv_recall_mean = cv_results["test_recall"].mean()
    cv_recall_std = cv_results["test_recall"].std()
    cv_f1_mean = cv_results["test_f1"].mean()
    cv_f1_std = cv_results["test_f1"].std()

    print(f"CV Accuracy:  {cv_accuracy_mean:.4f} (+/- {cv_accuracy_std:.4f})")
    print(f"CV Precision: {cv_precision_mean:.4f} (+/- {cv_precision_std:.4f})")
    print(f"CV Recall:    {cv_recall_mean:.4f} (+/- {cv_recall_std:.4f})")
    print(f"CV F1 Score:  {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")

    # 6. Save model + preprocessor so we don't need to retrain every time
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(preprocessor, "spam_preprocessor.pkl")

    # 7. Save metrics/metadata so the dashboard (app.py) can display them
    #    without needing to retrain or recompute anything
    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_samples": int(len(df)),
        "spam_count": int(df["label"].sum()),
        "ham_count": int((df["label"] == 0).sum()),
        # Cross-validated metrics -> the headline numbers shown on the dashboard
        "cv_accuracy_mean": round(float(cv_accuracy_mean), 4),
        "cv_accuracy_std": round(float(cv_accuracy_std), 4),
        "cv_precision_mean": round(float(cv_precision_mean), 4),
        "cv_precision_std": round(float(cv_precision_std), 4),
        "cv_recall_mean": round(float(cv_recall_mean), 4),
        "cv_recall_std": round(float(cv_recall_std), 4),
        "cv_f1_mean": round(float(cv_f1_mean), 4),
        "cv_f1_std": round(float(cv_f1_std), 4),
        # Single held-out test split -> shown only as a small secondary caption
        "test_accuracy": round(float(acc), 4),
        "test_precision": round(float(precision), 4),
        "test_recall": round(float(recall), 4),
        "test_f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "model_type": "Multinomial Naive Bayes",
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved spam_model.pkl, spam_preprocessor.pkl, and metrics.json")

    return model, preprocessor


if __name__ == "__main__":
    train_model("spam.csv")