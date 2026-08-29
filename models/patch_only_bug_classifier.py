import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
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


def prepare_dataset(records):

    texts = []
    labels = []

    for record in records:

        patch = record.get(
            "patch",
            ""
        )

        if patch is None:
            patch = ""

        texts.append(
            patch
        )

        labels.append(
            record["label"]
        )

    return texts, labels


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
        "\nLoading patch-only dataset..."
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
        train_labels
    ) = prepare_dataset(
        train_records
    )

    (
        val_texts,
        val_labels
    ) = prepare_dataset(
        val_records
    )

    (
        test_texts,
        test_labels
    ) = prepare_dataset(
        test_records
    )

    print(
        f"Train examples: "
        f"{len(train_texts)}"
    )

    print(
        f"Validation examples: "
        f"{len(val_texts)}"
    )

    print(
        f"Test examples: "
        f"{len(test_texts)}"
    )

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

    train_features = (
        vectorizer.fit_transform(
            train_texts
        )
    )

    val_features = (
        vectorizer.transform(
            val_texts
        )
    )

    test_features = (
        vectorizer.transform(
            test_texts
        )
    )

    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        "\nTraining patch-only model..."
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    model.fit(
        train_features,
        train_labels
    )

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