import json
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from tokenizers import Tokenizer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from models.hybrid_transformer_bug_classifier import (
    HybridTransformerBugClassifier,
)


# ============================================================
# PATHS
# ============================================================

TEST_PATH = (
    "data/processed/"
    "classification_splits/test.jsonl"
)

TOKENIZER_PATH = (
    "data/processed/"
    "code_tokenizer/tokenizer.json"
)

CHECKPOINT_PATH = Path(
    "checkpoints/"
    "hybrid_transformer_bug_classifier.pt"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "hybrid_test_predictions.jsonl"
)


# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH = 256

BATCH_SIZE = 8

VOCAB_SIZE = 16000

EMBEDDING_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 3

FEEDFORWARD_DIM = 256

DROPOUT = 0.25

NUMERIC_FEATURE_COUNT = 4

NUM_CLASSES = 2


# ============================================================
# JSONL LOADER
# ============================================================

def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} in {path}: "
                    f"{exc}"
                ) from exc

            records.append(
                record
            )

    if not records:

        raise ValueError(
            f"No records found in {path}"
        )

    return records


# ============================================================
# NUMERIC FEATURES
# ============================================================

def get_numeric_features(record):
    """
    Extract the exact four numeric features
    used during training.

    Features:

    1. added lines
    2. removed lines
    3. total changed lines
    4. added - removed
    """

    diff_stats = record.get(
        "diff_stats",
        {},
    )

    if not isinstance(
        diff_stats,
        dict,
    ):

        diff_stats = {}

    # --------------------------------------------------------
    # Added lines
    # --------------------------------------------------------

    added = record.get(
        "added_line_count"
    )

    if added is None:

        added = diff_stats.get(
            "added_lines",
            0,
        )

    # --------------------------------------------------------
    # Removed lines
    # --------------------------------------------------------

    removed = record.get(
        "removed_line_count"
    )

    if removed is None:

        removed = diff_stats.get(
            "deleted_lines",
            0,
        )

    # --------------------------------------------------------
    # Convert safely
    # --------------------------------------------------------

    try:

        added = float(
            added or 0
        )

    except (
        TypeError,
        ValueError,
    ):

        added = 0.0

    try:

        removed = float(
            removed or 0
        )

    except (
        TypeError,
        ValueError,
    ):

        removed = 0.0

    # --------------------------------------------------------
    # Derived features
    # --------------------------------------------------------

    total = (
        added
        + removed
    )

    balance = (
        added
        - removed
    )

    return [
        added,
        removed,
        total,
        balance,
    ]


# ============================================================
# BUILD NUMERIC MATRIX
# ============================================================

def build_numeric_matrix(
    records
):

    rows = [
        get_numeric_features(
            record
        )
        for record in records
    ]

    return np.asarray(
        rows,
        dtype=np.float32,
    )


# ============================================================
# SCALER FROM CHECKPOINT
# ============================================================

def build_scaler_from_checkpoint(
    checkpoint
):

    if (
        "scaler_mean"
        not in checkpoint
    ):

        raise ValueError(
            "Checkpoint does not contain "
            "'scaler_mean'. "
            "This checkpoint cannot reproduce "
            "the training preprocessing."
        )

    if (
        "scaler_scale"
        not in checkpoint
    ):

        raise ValueError(
            "Checkpoint does not contain "
            "'scaler_scale'. "
            "This checkpoint cannot reproduce "
            "the training preprocessing."
        )

    scaler = StandardScaler()

    scaler.mean_ = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float64,
    )

    scaler.scale_ = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float64,
    )

    scaler.var_ = (
        scaler.scale_
        ** 2
    )

    scaler.n_features_in_ = (
        len(
            scaler.mean_
        )
    )

    return scaler


# ============================================================
# DATASET
# ============================================================

class HybridPredictionDataset(
    Dataset
):

    def __init__(
        self,
        records,
        tokenizer,
        scaler,
    ):

        self.records = records

        self.tokenizer = tokenizer

        self.scaler = scaler

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        vocab = (
            tokenizer.get_vocab()
        )

        required_tokens = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
        ]

        for token in required_tokens:

            if token not in vocab:

                raise ValueError(
                    f"Tokenizer is missing "
                    f"required token: {token}"
                )

        self.pad_id = vocab[
            "<PAD>"
        ]

        self.bos_id = vocab[
            "<BOS>"
        ]

        self.eos_id = vocab[
            "<EOS>"
        ]

        # ----------------------------------------------------
        # Numeric features
        # ----------------------------------------------------

        numeric_rows = (
            build_numeric_matrix(
                records
            )
        )

        self.numeric_features = (
            scaler.transform(
                numeric_rows
            )
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(
        self
    ):

        return len(
            self.records
        )

    # ========================================================
    # ENCODE PATCH
    # ========================================================

    def encode_patch(
        self,
        patch,
    ):

        if patch is None:

            patch = ""

        patch = str(
            patch
        )

        ids = (
            self.tokenizer
            .encode(patch)
            .ids
        )

        # ----------------------------------------------------
        # Add BOS/EOS
        # ----------------------------------------------------

        ids = (
            [self.bos_id]
            + ids
            + [self.eos_id]
        )

        # ----------------------------------------------------
        # Truncate
        # ----------------------------------------------------

        if (
            len(ids)
            > MAX_LENGTH
        ):

            ids = ids[
                :MAX_LENGTH
            ]

            ids[-1] = (
                self.eos_id
            )

        # ----------------------------------------------------
        # Attention mask
        # ----------------------------------------------------

        mask = [
            1
        ] * len(ids)

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        padding = (
            MAX_LENGTH
            - len(ids)
        )

        if padding > 0:

            ids.extend(
                [
                    self.pad_id
                ]
                * padding
            )

            mask.extend(
                [
                    0
                ]
                * padding
            )

        return (
            ids,
            mask,
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        record = (
            self.records[
                index
            ]
        )

        ids, mask = (
            self.encode_patch(
                record.get(
                    "patch",
                    "",
                )
            )
        )

        numeric = (
            self.numeric_features[
                index
            ]
        )

        label = int(
            record["label"]
        )

        return {

            "input_ids":
                torch.tensor(
                    ids,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    mask,
                    dtype=torch.long,
                ),

            "numeric_features":
                torch.tensor(
                    numeric,
                    dtype=torch.float32,
                ),

            "label":
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),

            "index":
                index,
        }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
):

    return {

        "accuracy":
            accuracy_score(
                labels,
                predictions,
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print(
        "RepoMind Hybrid Transformer Prediction"
    )
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [
        Path(TEST_PATH),
        Path(TOKENIZER_PATH),
        CHECKPOINT_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: "
                f"{path}"
            )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print()
    print(
        "Loading test data..."
    )

    test_records = (
        load_jsonl(
            TEST_PATH
        )
    )

    print(
        "Test records:",
        len(test_records),
    )

    # ========================================================
    # EXPECTED TEST SIZE
    # ========================================================

    if len(test_records) != 1883:

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected the training test split "
            "to contain 1883 records."
        )

        print(
            f"Current test records: "
            f"{len(test_records)}"
        )

        print(
            "Make sure you are using:"
        )

        print(
            TEST_PATH
        )

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        Tokenizer.from_file(
            TOKENIZER_PATH
        )
    )

    tokenizer_vocab_size = (
        tokenizer.get_vocab_size()
    )

    print(
        "Vocabulary size:",
        tokenizer_vocab_size,
    )

    if (
        tokenizer_vocab_size
        != VOCAB_SIZE
    ):

        raise ValueError(
            f"VOCAB_SIZE={VOCAB_SIZE}, "
            f"but tokenizer has "
            f"{tokenizer_vocab_size} tokens."
        )

    vocab = (
        tokenizer.get_vocab()
    )

    print(
        "<PAD>:",
        vocab.get(
            "<PAD>"
        ),
    )

    print(
        "<BOS>:",
        vocab.get(
            "<BOS>"
        ),
    )

    print(
        "<EOS>:",
        vocab.get(
            "<EOS>"
        ),
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print(
        "Loading checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise ValueError(
            "Checkpoint must be a dictionary."
        )

    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise ValueError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    # ========================================================
    # SCALER
    #
    # IMPORTANT:
    # Use the EXACT scaler saved during training.
    # ========================================================

    scaler = (
        build_scaler_from_checkpoint(
            checkpoint
        )
    )

    print()
    print(
        "Scaler loaded from checkpoint."
    )

    print(
        "Scaler mean:",
        scaler.mean_,
    )

    print(
        "Scaler scale:",
        scaler.scale_,
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = (
        HybridPredictionDataset(
            records=test_records,
            tokenizer=tokenizer,
            scaler=scaler,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print()
    print(
        "Test batches:",
        len(loader),
    )

    # ========================================================
    # MODEL CONFIG
    # ========================================================

    checkpoint_config = (
        checkpoint.get(
            "config",
            {}
        )
    )

    if checkpoint_config:

        print()
        print(
            "Checkpoint configuration:"
        )

        for key, value in (
            checkpoint_config.items()
        ):

            print(
                f"  {key}: {value}"
            )

    # --------------------------------------------------------
    # Use checkpoint config where available
    # --------------------------------------------------------

    model_vocab_size = (
        checkpoint_config.get(
            "vocab_size",
            VOCAB_SIZE,
        )
    )

    model_max_length = (
        checkpoint_config.get(
            "max_length",
            MAX_LENGTH,
        )
    )

    model_embedding_dim = (
        checkpoint_config.get(
            "embedding_dim",
            EMBEDDING_DIM,
        )
    )

    model_num_heads = (
        checkpoint_config.get(
            "num_heads",
            NUM_HEADS,
        )
    )

    model_num_layers = (
        checkpoint_config.get(
            "num_layers",
            NUM_LAYERS,
        )
    )

    model_feedforward_dim = (
        checkpoint_config.get(
            "feedforward_dim",
            FEEDFORWARD_DIM,
        )
    )

    model_dropout = (
        checkpoint_config.get(
            "dropout",
            DROPOUT,
        )
    )

    model_numeric_count = (
        checkpoint_config.get(
            "numeric_feature_count",
            NUMERIC_FEATURE_COUNT,
        )
    )

    model_num_classes = (
        checkpoint_config.get(
            "num_classes",
            NUM_CLASSES,
        )
    )

    # ========================================================
    # VERIFY CONFIG
    # ========================================================

    if (
        model_vocab_size
        != tokenizer_vocab_size
    ):

        raise ValueError(
            "Checkpoint vocabulary size "
            "does not match tokenizer vocabulary size."
        )

    if (
        model_max_length
        != MAX_LENGTH
    ):

        raise ValueError(
            f"Checkpoint max_length="
            f"{model_max_length}, "
            f"but prediction MAX_LENGTH="
            f"{MAX_LENGTH}."
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = (
        HybridTransformerBugClassifier(

            vocab_size=
                model_vocab_size,

            max_length=
                model_max_length,

            embedding_dim=
                model_embedding_dim,

            num_heads=
                model_num_heads,

            num_layers=
                model_num_layers,

            feedforward_dim=
                model_feedforward_dim,

            dropout=
                model_dropout,

            numeric_feature_count=
                model_numeric_count,

            num_classes=
                model_num_classes,

            pad_token_id=
                vocab["<PAD>"],
        )
    )

    model = model.to(
        device
    )

    # ========================================================
    # LOAD MODEL WEIGHTS
    # ========================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print()
    print(
        "Checkpoint loaded:",
        CHECKPOINT_PATH,
    )

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"],
        )

    if "best_val_f1" in checkpoint:

        print(
            "Best validation F1:",
            f"{checkpoint['best_val_f1']:.4f}",
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    all_labels = []

    all_predictions = []

    all_probabilities = []

    all_indices = []

    print()
    print(
        "Running predictions..."
    )

    with torch.no_grad():

        for batch in loader:

            # ------------------------------------------------
            # Input IDs
            # ------------------------------------------------

            input_ids = (
                batch[
                    "input_ids"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # Attention mask
            # ------------------------------------------------

            attention_mask = (
                batch[
                    "attention_mask"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # Numeric features
            # ------------------------------------------------

            numeric_features = (
                batch[
                    "numeric_features"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            labels = (
                batch[
                    "label"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                input_ids,
                attention_mask,
                numeric_features,
            )

            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = (
                torch.argmax(
                    logits,
                    dim=1,
                )
            )

            # ------------------------------------------------
            # Collect
            # ------------------------------------------------

            all_labels.extend(
                labels
                .cpu()
                .tolist()
            )

            all_predictions.extend(
                prediction
                .cpu()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .tolist()
            )

            all_indices.extend(
                batch[
                    "index"
                ]
                .tolist()
            )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = (
        calculate_metrics(
            all_labels,
            all_predictions,
        )
    )

    print()
    print("=" * 60)
    print(
        "TEST METRICS"
    )
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{metrics['f1']:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    print()
    print(
        "Confusion Matrix"
    )
    print(
        "----------------"
    )

    print(
        matrix
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print(
        "Classification Report"
    )
    print(
        "----------------------"
    )

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=[
                "non_bug_fix",
                "bug_fix",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        for (
            index,
            label,
            prediction,
            probabilities,
        ) in zip(
            all_indices,
            all_labels,
            all_predictions,
            all_probabilities,
        ):

            record = (
                test_records[
                    index
                ]
            )

            output = {

                "record_id":
                    record.get(
                        "record_id"
                    ),

                "repo_id":
                    record.get(
                        "repo_id"
                    ),

                "commit":
                    record.get(
                        "commit"
                    ),

                "true_label":
                    int(
                        label
                    ),

                "predicted_label":
                    int(
                        prediction
                    ),

                "probability_non_bug_fix":
                    float(
                        probabilities[0]
                    ),

                "probability_bug_fix":
                    float(
                        probabilities[1]
                    ),

                "correct":
                    bool(
                        label
                        == prediction
                    ),
            }

            file.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "Predictions saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 60)
    print(
        "PREDICTION COMPLETE"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()