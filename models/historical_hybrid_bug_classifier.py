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
    "data/processed/"
    "historical_bug_classifier_splits/train.json"
)

VAL_PATH = (
    "data/processed/"
    "historical_bug_classifier_splits/validation.json"
)

TEST_PATH = (
    "data/processed/"
    "historical_bug_classifier_splits/test.json"
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def get_extension(file_path):
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

        # ==========================================
        # PATCH TEXT
        # ==========================================

        patch = record.get(
            "patch",
            ""
        )

        if patch is None:
            patch = ""

        patch_texts.append(
            patch
        )

        # ==========================================
        # CURRENT CHANGE FEATURES
        # ==========================================

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

        # ==========================================
        # HISTORICAL FEATURES
        # ==========================================

        history = record.get(
            "historical_features",
            {}
        )

        prior_commit_count = history.get(
            "prior_commit_count",
            0
        )

        prior_bug_fix_count = history.get(
            "prior_bug_fix_count",
            0
        )

        prior_total_insertions = history.get(
            "prior_total_insertions",
            0
        )

        prior_total_deletions = history.get(
            "prior_total_deletions",
            0
        )

        prior_total_churn = history.get(
            "prior_total_churn",
            0
        )

        prior_unique_authors = history.get(
            "prior_unique_authors",
            0
        )

        days_since_last_change = history.get(
            "days_since_last_change",
            -1
        )

        historical_bug_fix_ratio = history.get(
            "historical_bug_fix_ratio",
            0.0
        )

        average_prior_churn = history.get(
            "average_prior_churn",
            0.0
        )

        # ==========================================
        # COMBINE NUMERIC FEATURES
        # ==========================================

        numeric_features.append(
            [
                added,
                removed,
                total_changes,
                change_balance,

                prior_commit_count,
                prior_bug_fix_count,
                prior_total_insertions,
                prior_total_deletions,
                prior_total_churn,
                prior_unique_authors,
                days_since_last_change,
                historical_bug_fix_ratio,
                average_prior_churn,
            ]
        )

        # ==========================================
        # CATEGORICAL FEATURES
        # ==========================================

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
        "\nLoading historical datasets..."
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

    # ==========================================
    # TEXT FEATURES
    # ==========================================

    print(
        "\nBuilding patch TF-IDF features..."
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

    # ==========================================
    # NUMERIC FEATURES
    # ==========================================

    print(
        "Scaling numeric + historical features..."
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

    # ==========================================
    # CATEGORICAL FEATURES
    # ==========================================

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

    # ==========================================
    # COMBINE
    # ==========================================

    print(
        "Combining text + metadata + history..."
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

    # ==========================================
    # MODEL
    # ==========================================

    print(
        "\nTraining Historical Hybrid Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        C=1.0
    )

    model.fit(
        train_features,
        train_labels
    )

    # ==========================================
    # EVALUATION
    # ==========================================

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