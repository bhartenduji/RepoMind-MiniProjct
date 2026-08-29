import json
import numpy as np

from sklearn.preprocessing import StandardScaler
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


HISTORY_FEATURE_NAMES = [
    "prior_commit_count",
    "prior_bug_fix_count",
    "prior_total_insertions",
    "prior_total_deletions",
    "prior_total_churn",
    "prior_unique_authors",
    "days_since_last_change",
    "historical_bug_fix_ratio",
    "average_prior_churn",
]


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def prepare_dataset(records):
    features = []
    labels = []

    for record in records:

        history = record.get(
            "historical_features",
            {}
        )

        feature_row = []

        for feature_name in HISTORY_FEATURE_NAMES:

            value = history.get(
                feature_name,
                0
            )

            # Safety fallback
            if value is None:
                value = 0

            feature_row.append(
                float(value)
            )

        features.append(
            feature_row
        )

        labels.append(
            record["label"]
        )

    return (
        np.array(
            features,
            dtype=float
        ),
        np.array(
            labels
        )
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

    probabilities = model.predict_proba(
        features
    )[:, 1]

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

    print(
        "\nFirst 10 bug probabilities:"
    )

    for probability in probabilities[:10]:

        print(
            f"{probability:.4f}"
        )


def print_model_coefficients(
    model
):
    """
    Show which historical features push
    predictions toward bug-fix or normal-change.
    """

    coefficients = model.coef_[0]

    feature_importance = list(
        zip(
            HISTORY_FEATURE_NAMES,
            coefficients
        )
    )

    feature_importance.sort(
        key=lambda item:
        abs(item[1]),
        reverse=True
    )

    print(
        "\nHistorical Feature Weights"
    )

    print(
        "-------------------------"
    )

    for (
        feature_name,
        coefficient
    ) in feature_importance:

        direction = (
            "bug-risk"
            if coefficient > 0
            else "normal-change"
        )

        print(
            f"{feature_name:30s} "
            f"{coefficient: .4f} "
            f"-> {direction}"
        )


if __name__ == "__main__":

    print(
        "\nLoading historical splits..."
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
        train_features,
        train_labels
    ) = prepare_dataset(
        train_records
    )

    (
        val_features,
        val_labels
    ) = prepare_dataset(
        val_records
    )

    (
        test_features,
        test_labels
    ) = prepare_dataset(
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

    print(
        f"Historical feature count: "
        f"{train_features.shape[1]}"
    )

    print(
        "\nScaling historical features..."
    )

    scaler = StandardScaler()

    train_scaled = (
        scaler.fit_transform(
            train_features
        )
    )

    val_scaled = (
        scaler.transform(
            val_features
        )
    )

    test_scaled = (
        scaler.transform(
            test_features
        )
    )

    print(
        "\nTraining history-only classifier..."
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0
    )

    model.fit(
        train_scaled,
        train_labels
    )

    print_model_coefficients(
        model
    )

    evaluate(
        model,
        val_scaled,
        val_labels,
        "Validation"
    )

    evaluate(
        model,
        test_scaled,
        test_labels,
        "Test"
    )