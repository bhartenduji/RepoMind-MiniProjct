import json
from pathlib import Path

import numpy as np

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


TRAIN_PATH = (
    "data/processed/bug_classifier_splits/train.json"
)

VAL_PATH = (
    "data/processed/bug_classifier_splits/validation.json"
)

TEST_PATH = (
    "data/processed/bug_classifier_splits/test.json"
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def get_extension(file_path):
    """
    Example:

    src/flask/app.py
        -> .py
    """

    if not file_path:
        return "unknown"

    suffix = Path(
        file_path
    ).suffix.lower()

    if not suffix:
        return "no_extension"

    return suffix


def prepare_records(records):

    patch_texts = []

    numeric_features = []

    categorical_features = []

    labels = []

    for record in records:

        # =========================================
        # TEXT
        # =========================================

        patch = record.get(
            "patch",
            ""
        )

        if patch is None:
            patch = ""

        patch_texts.append(
            patch
        )

        # =========================================
        # NUMERIC
        # =========================================

        added = record.get(
            "added_line_count",
            0
        )

        removed = record.get(
            "removed_line_count",
            0
        )

        total_changes = (
            added
            +
            removed
        )

        change_balance = (
            added
            -
            removed
        )

        numeric_features.append(
            [
                added,
                removed,
                total_changes,
                change_balance,
            ]
        )

        # =========================================
        # CATEGORICAL
        # =========================================

        change_type = record.get(
            "change_type",
            "unknown"
        )

        extension = get_extension(
            record.get(
                "file_path"
            )
        )

        categorical_features.append(
            [
                change_type,
                extension,
            ]
        )

        labels.append(
            record["label"]
        )

    return (
        patch_texts,
        np.array(
            numeric_features,
            dtype=float
        ),
        categorical_features,
        np.array(
            labels
        ),
    )


def evaluate(
    model,
    features,
    labels,
    dataset_name
):

    predictions = model.predict(
        features
    )

    print(
        f"\n{dataset_name} Results"
    )

    print(
        "-------------------------"
    )

    print(
        f"Accuracy: "
        f"{accuracy_score(labels, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(labels, predictions):.4f}"
    )

    print(
        f"Recall: "
        f"{recall_score(labels, predictions):.4f}"
    )

    print(
        f"F1 Score: "
        f"{f1_score(labels, predictions):.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            labels,
            predictions
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            labels,
            predictions
        )
    )


if __name__ == "__main__":

    print(
        "\nLoading datasets..."
    )

    train_records = load_json(
        TRAIN_PATH
    )

    val_records = load_json(
        VAL_PATH
    )

    test_records = load_json(
        TEST_PATH
    )

    (
        train_texts,
        train_numeric,
        train_categorical,
        train_labels,
    ) = prepare_records(
        train_records
    )

    (
        val_texts,
        val_numeric,
        val_categorical,
        val_labels,
    ) = prepare_records(
        val_records
    )

    (
        test_texts,
        test_numeric,
        test_categorical,
        test_labels,
    ) = prepare_records(
        test_records
    )

    print(
        f"Train examples: "
        f"{len(train_labels)}"
    )

    print(
        f"Validation examples: "
        f"{len(val_labels)}"
    )

    print(
        f"Test examples: "
        f"{len(test_labels)}"
    )

    # =====================================================
    # TEXT FEATURES
    # =====================================================

    print(
        "\nBuilding TF-IDF patch features..."
    )

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b"
    )

    train_text_features = (
        vectorizer.fit_transform(
            train_texts
        )
    )

    val_text_features = (
        vectorizer.transform(
            val_texts
        )
    )

    test_text_features = (
        vectorizer.transform(
            test_texts
        )
    )

    print(
        f"Text vocabulary: "
        f"{len(vectorizer.vocabulary_)}"
    )

    # =====================================================
    # NUMERIC FEATURES
    # =====================================================

    print(
        "Scaling numeric features..."
    )

    scaler = StandardScaler()

    train_numeric_scaled = (
        scaler.fit_transform(
            train_numeric
        )
    )

    val_numeric_scaled = (
        scaler.transform(
            val_numeric
        )
    )

    test_numeric_scaled = (
        scaler.transform(
            test_numeric
        )
    )

    # =====================================================
    # CATEGORICAL FEATURES
    # =====================================================

    print(
        "Encoding categorical features..."
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    train_categorical_encoded = (
        encoder.fit_transform(
            train_categorical
        )
    )

    val_categorical_encoded = (
        encoder.transform(
            val_categorical
        )
    )

    test_categorical_encoded = (
        encoder.transform(
            test_categorical
        )
    )

    # =====================================================
    # COMBINE FEATURES
    # =====================================================

    print(
        "Combining features..."
    )

    train_features = hstack(
        [
            train_text_features,
            train_numeric_scaled,
            train_categorical_encoded,
        ]
    ).tocsr()

    val_features = hstack(
        [
            val_text_features,
            val_numeric_scaled,
            val_categorical_encoded,
        ]
    ).tocsr()

    test_features = hstack(
        [
            test_text_features,
            test_numeric_scaled,
            test_categorical_encoded,
        ]
    ).tocsr()

    print(
        f"Final feature count: "
        f"{train_features.shape[1]}"
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    print(
        "\nTraining Hybrid Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0
    )

    model.fit(
        train_features,
        train_labels
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    evaluate(
        model,
        val_features,
        val_labels,
        "Validation"
    )

    evaluate(
        model,
        test_features,
        test_labels,
        "Test"
    )