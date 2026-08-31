import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from models.hybrid_transformer_bug_classifier import (
    HybridTransformerBugClassifier,
)


CHECKPOINT_PATH = (
    "checkpoints/"
    "hybrid_transformer_bug_classifier_augmented_500.pt"
)

TOKENIZER_PATH = (
    "data/processed/"
    "code_tokenizer/tokenizer.json"
)
BUG_FIX_THRESHOLD = 0.51


def get_numeric_features(record):

    diff_stats = record.get(
        "diff_stats",
        {}
    )

    if not isinstance(diff_stats, dict):
        diff_stats = {}

    added = record.get(
        "added_line_count"
    )

    if added is None:
        added = diff_stats.get(
            "added_lines",
            0
        )

    removed = record.get(
        "removed_line_count"
    )

    if removed is None:
        removed = diff_stats.get(
            "deleted_lines",
            0
        )

    files_changed = diff_stats.get(
        "files_changed",
        0
    )

    try:
        added = float(added or 0)
    except (TypeError, ValueError):
        added = 0.0

    try:
        removed = float(removed or 0)
    except (TypeError, ValueError):
        removed = 0.0

    try:
        files_changed = float(
            files_changed or 0
        )
    except (TypeError, ValueError):
        files_changed = 0.0

    total = (
        added
        + removed
    )

    balance = (
        added
        - removed
    )

    changed_files = record.get(
        "changed_files",
        []
    )

    if not isinstance(
        changed_files,
        list
    ):
        changed_files = []

    actual_file_count = float(
        len(changed_files)
    )

    source_files = 0.0
    test_files = 0.0
    ignored_files = 0.0

    for file_info in changed_files:

        if not isinstance(
            file_info,
            dict
        ):
            continue

        if file_info.get(
            "is_source",
            False
        ):
            source_files += 1.0

        path = str(
            file_info.get(
                "path",
                ""
            )
        ).lower()

        if (
            "/test/" in path
            or path.startswith("test/")
            or "/tests/" in path
            or path.startswith("tests/")
            or path.startswith("test_")
            or path.endswith("_test.py")
        ):
            test_files += 1.0

        if file_info.get(
            "ignored",
            False
        ):
            ignored_files += 1.0

    if files_changed <= 0:
        files_changed = actual_file_count

    denominator = max(
        files_changed,
        1.0
    )

    avg_added = (
        added
        / denominator
    )

    avg_removed = (
        removed
        / denominator
    )

    avg_changed = (
        total
        / denominator
    )

    source_ratio = (
        source_files
        / denominator
    )

    test_ratio = (
        test_files
        / denominator
    )

    patch = record.get(
        "patch",
        ""
    )

    if patch is None:
        patch = ""

    if not isinstance(
        patch,
        str
    ):
        patch = str(patch)

    patch_length = float(
        len(patch)
    )

    patch_line_count = float(
        patch.count("\n") + 1
        if patch
        else 0
    )

    features = [
        added,
        removed,
        total,
        balance,
        files_changed,
        source_files,
        test_files,
        ignored_files,
        avg_added,
        avg_removed,
        avg_changed,
        source_ratio,
        test_ratio,
        patch_length,
        patch_line_count,
    ]

    if len(features) != 15:
        raise ValueError(
            f"Expected 15 numeric features, "
            f"got {len(features)}"
        )

    return features


def encode_patch(
    tokenizer,
    patch,
    bos_id,
    eos_id,
    pad_id,
    max_length,
):

    if patch is None:
        patch = ""

    patch = str(patch)

    ids = tokenizer.encode(
        patch
    ).ids

    ids = (
        [bos_id]
        +
        ids
        +
        [eos_id]
    )

    if len(ids) > max_length:

        content_length = (
            max_length - 2
        )

        head_length = (
            content_length // 2
        )

        tail_length = (
            content_length - head_length
        )

        content = ids[1:-1]

        ids = (
            [bos_id]
            +
            content[:head_length]
            +
            content[-tail_length:]
            +
            [eos_id]
        )

    mask = (
        [1]
        *
        len(ids)
    )

    padding = (
        max_length
        - len(ids)
    )

    if padding > 0:

        ids.extend(
            [pad_id]
            *
            padding
        )

        mask.extend(
            [0]
            *
            padding
        )

    return ids, mask


def load_model():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    config = checkpoint["config"]

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    vocab = tokenizer.get_vocab()

    required_tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
    ]

    for token in required_tokens:

        if token not in vocab:
            raise ValueError(
                f"Tokenizer missing "
                f"required token: {token}"
            )

    model = (
        HybridTransformerBugClassifier(
            vocab_size=config[
                "vocab_size"
            ],
            max_length=config[
                "max_length"
            ],
            embedding_dim=config[
                "embedding_dim"
            ],
            num_heads=config[
                "num_heads"
            ],
            num_layers=config[
                "num_layers"
            ],
            feedforward_dim=config[
                "feedforward_dim"
            ],
            dropout=config[
                "dropout"
            ],
            numeric_feature_count=config[
                "numeric_feature_count"
            ],
            num_classes=config[
                "num_classes"
            ],
            pad_token_id=vocab[
                "<PAD>"
            ],
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        tokenizer,
        checkpoint,
    )


def predict(record, model, tokenizer, checkpoint):

    config = checkpoint["config"]

    bos_id = tokenizer.get_vocab()[
        "<BOS>"
    ]

    eos_id = tokenizer.get_vocab()[
        "<EOS>"
    ]

    pad_id = tokenizer.get_vocab()[
        "<PAD>"
    ]

    ids, mask = encode_patch(
        tokenizer=tokenizer,
        patch=record.get(
            "patch",
            ""
        ),
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_length=config[
            "max_length"
        ],
    )

    numeric = np.asarray(
        get_numeric_features(record),
        dtype=np.float32,
    )

    mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float32,
    )

    numeric = (
        (numeric - mean)
        / scale
    )

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [mask],
        dtype=torch.long,
    )

    numeric_features = torch.tensor(
        [numeric],
        dtype=torch.float32,
    )

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_features=numeric_features,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        prediction = 1 if probabilities[1].item() >= BUG_FIX_THRESHOLD else 0

    labels = [
        "non_bug_fix",
        "bug_fix",
    ]

    return {
        "prediction": labels[
            prediction
        ],
        "non_bug_fix_probability": float(
            probabilities[0].item()
        ),
        "bug_fix_probability": float(
            probabilities[1].item()
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "RepoMind hybrid transformer "
            "bug-fix classifier"
        )
    )

    parser.add_argument(
        "json_file",
        help=(
            "JSON file containing "
            "one record"
        ),
    )

    args = parser.parse_args()

    path = Path(
        args.json_file
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        record = json.load(f)

    model, tokenizer, checkpoint = load_model()

    result = predict(
        record,
        model,
        tokenizer,
        checkpoint,
)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
