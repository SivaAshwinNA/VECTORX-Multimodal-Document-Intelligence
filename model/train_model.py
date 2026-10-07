from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# VECTORX EVIDENCE RELEVANCE MODEL
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "training_data.csv"
MODEL_DIR = ROOT / "model" / "saved_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("VECTORX EVIDENCE RELEVANCE MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n[1/6] Loading training dataset...")

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "question",
        "evidence",
        "label"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in training dataset: {missing}"
        )

    print(f"Training examples: {len(df)}")

    # --------------------------------------------------------
    # 2. Combine question + evidence
    # --------------------------------------------------------

    print("\n[2/6] Preparing training examples...")

    df["combined_text"] = (
        "question: "
        + df["question"].astype(str)
        + " evidence: "
        + df["evidence"].astype(str)
    )

    X = df["combined_text"]
    y = df["label"]

    # --------------------------------------------------------
    # 3. Split dataset
    # --------------------------------------------------------

    print("\n[3/6] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # --------------------------------------------------------
    # 4. TF-IDF
    # --------------------------------------------------------

    print("\n[4/6] Building TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print(
        f"Feature matrix: "
        f"{X_train_vectorized.shape}"
    )

    # --------------------------------------------------------
    # 5. Train classifier
    # --------------------------------------------------------

    print("\n[5/6] Training VECTORX model...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_vectorized,
        y_train
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    print("\n[6/6] Evaluating model...")

    predictions = model.predict(
        X_test_vectorized
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = MODEL_DIR / "evidence_relevance_model.joblib"
    vectorizer_file = MODEL_DIR / "tfidf_vectorizer.joblib"

    joblib.dump(
        model,
        model_file
    )

    joblib.dump(
        vectorizer,
        vectorizer_file
    )

    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(f"\nModel:")
    print(model_file)

    print(f"\nVectorizer:")
    print(vectorizer_file)

    print("\nVECTORX MODEL TRAINING COMPLETE.")


if __name__ == "__main__":
    main()