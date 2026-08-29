import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

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
import random
import numpy as np
import torch


# =========================================================
# PATHS
# =========================================================

TRAIN_PATH = (
    "data/processed/"
    "classification_splits/train_augmented_500.jsonl"
)

VAL_PATH = (
    "data/processed/"
    "classification_splits/validation.jsonl"
)

TEST_PATH = (
    "data/processed/"
    "classification_splits/test.jsonl"
)

TOKENIZER_PATH = (
    "data/processed/"
    "code_tokenizer/tokenizer.json"
)

CHECKPOINT_DIR = Path(
    "checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "hybrid_transformer_bug_classifier_augmented_500.pt"
)


# =========================================================
# CONFIG
# =========================================================

MAX_LENGTH = 1024

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.02

GRAD_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 2

NUMERIC_FEATURE_COUNT = 15


# =========================================================
# MODEL CONFIG
# =========================================================

VOCAB_SIZE = 16000

EMBEDDING_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 3

FEEDFORWARD_DIM = 256

DROPOUT = 0.25

NUM_CLASSES = 2


# =========================================================
# RANDOM SEED
# =========================================================

SEED = 42


def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# JSONL LOADER
# =========================================================

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

                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} in {path}: "
                    f"{exc}"
                ) from exc

            records.append(record)

    if not records:

        raise ValueError(
            f"No records found in {path}"
        )

    return records


# =========================================================
# NUMERIC FEATURES
# =========================================================

def get_numeric_features(record):
    """
    Extract 15 numeric features.
    """

    diff_stats = record.get(
        "diff_stats",
        {}
    )

    if not isinstance(diff_stats, dict):
        diff_stats = {}

    # -----------------------------------------------------
    # DIFF STATISTICS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CHANGED FILES
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # FILE COUNT FALLBACK
    # -----------------------------------------------------

    if files_changed <= 0:
        files_changed = actual_file_count

    # -----------------------------------------------------
    # AVERAGES
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # PATCH FEATURES
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RETURN 15 FEATURES
    # -----------------------------------------------------

    return [
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

def build_numeric_matrix(records):

    rows = [
        get_numeric_features(record)
        for record in records
    ]

    return np.asarray(
        rows,
        dtype=np.float32,
    )


# =========================================================
# DATASET
# =========================================================

class HybridBugDataset(Dataset):

    def __init__(
        self,
        records,
        tokenizer,
        scaler,
        fit_scaler=False,
        max_length=MAX_LENGTH,
    ):

        self.records = records

        self.tokenizer = tokenizer

        self.scaler = scaler

        self.max_length = max_length

        vocab = tokenizer.get_vocab()

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

        numeric_rows = (
            build_numeric_matrix(
                records
            )
        )

        if fit_scaler:

            self.numeric_features = (
                scaler.fit_transform(
                    numeric_rows
                )
            )

        else:

            self.numeric_features = (
                scaler.transform(
                    numeric_rows
                )
            )

    def __len__(self):

        return len(
            self.records
        )

    def encode_patch(
        self,
        patch,
    ):

        if patch is None:
            patch = ""

        patch = str(patch)

        ids = self.tokenizer.encode(
            patch
        ).ids

        ids = (
            [self.bos_id]
            +
            ids
            +
            [self.eos_id]
        )

        if len(ids) > self.max_length:

            content_length = (
                self.max_length - 2
            )

            head_length = (
                content_length // 2
            )

            tail_length = (
                content_length - head_length
            )

            content = ids[1:-1]

            ids = (
                [self.bos_id]
                +
                content[:head_length]
                +
                content[-tail_length:]
                +
                [self.eos_id]
            )

        mask = (
            [1]
            *
            len(ids)
        )

        padding = (
            self.max_length
            -
            len(ids)
        )

        if padding > 0:

            ids.extend(
                [self.pad_id]
                *
                padding
            )

            mask.extend(
                [0]
                *
                padding
            )

        return ids, mask

    def __getitem__(
        self,
        index,
    ):

        record = self.records[
            index
        ]

        ids, mask = (
            self.encode_patch(
                record.get(
                    "patch",
                    ""
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
        }


# =========================================================
# DATA LOADERS
# =========================================================

def create_loader(
    records,
    tokenizer,
    scaler,
    fit_scaler,
    batch_size,
    shuffle,
):

    dataset = HybridBugDataset(
        records=records,
        tokenizer=tokenizer,
        scaler=scaler,
        fit_scaler=fit_scaler,
        max_length=MAX_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, loader


# =========================================================
# METRICS
# =========================================================

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


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):

    model.train()

    total_loss = 0.0

    labels_all = []

    predictions_all = []

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        numeric_features = batch[
            "numeric_features"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            ),
        ):

            logits = model(
                input_ids,
                attention_mask,
                numeric_features,
            )

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            loss.item()
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        labels_all.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        predictions_all.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        if batch_index % 50 == 0:

            print(
                f"Batch "
                f"{batch_index}/"
                f"{len(loader)} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )

    average_loss = (
        total_loss
        /
        max(len(loader), 1)
    )

    metrics = calculate_metrics(
        labels_all,
        predictions_all,
    )

    return (
        average_loss,
        metrics,
    )


# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    labels_all = []

    predictions_all = []

    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        numeric_features = batch[
            "numeric_features"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            ),
        ):

            logits = model(
                input_ids,
                attention_mask,
                numeric_features,
            )

            loss = criterion(
                logits,
                labels,
            )

        total_loss += (
            loss.item()
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        labels_all.extend(
            labels.cpu().tolist()
        )

        predictions_all.extend(
            predictions.cpu().tolist()
        )

    average_loss = (
        total_loss
        /
        max(len(loader), 1)
    )

    metrics = calculate_metrics(
        labels_all,
        predictions_all,
    )

    return (
        average_loss,
        metrics,
        labels_all,
        predictions_all,
    )


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(
    name,
    loss,
    metrics,
):

    print()

    print(
        name
    )

    print(
        "-------------------------"
    )

    print(
        f"Loss:      "
        f"{loss:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{metrics['f1']:.4f}"
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def print_distribution(
    name,
    records,
):

    counts = {
        0: 0,
        1: 0,
    }

    for record in records:

        label = int(
            record["label"]
        )

        counts[label] = (
            counts.get(label, 0)
            + 1
        )

    print(
        f"{name}: "
        f"total={len(records)} "
        f"| label_0={counts.get(0, 0)} "
        f"| label_1={counts.get(1, 0)}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed()

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()

    print(
        "RepoMind Neural Hybrid Transformer"
    )

    print(
        "==================================="
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    print()

    print(
        "Loading tokenizer..."
    )

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    tokenizer_vocab_size = (
        tokenizer.get_vocab_size()
    )

    print(
        f"Tokenizer vocabulary: "
        f"{tokenizer_vocab_size}"
    )

    if tokenizer_vocab_size != VOCAB_SIZE:

        raise ValueError(
            f"VOCAB_SIZE={VOCAB_SIZE}, "
            f"but tokenizer has "
            f"{tokenizer_vocab_size} tokens."
        )

    vocab = tokenizer.get_vocab()

    print(
        f"<PAD>: "
        f"{vocab.get('<PAD>')}"
    )

    print(
        f"<BOS>: "
        f"{vocab.get('<BOS>')}"
    )

    print(
        f"<EOS>: "
        f"{vocab.get('<EOS>')}"
    )

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    print()

    print(
        "Loading datasets..."
    )

    train_records = load_jsonl(
        TRAIN_PATH
    )

    val_records = load_jsonl(
        VAL_PATH
    )

    test_records = load_jsonl(
        TEST_PATH
    )

    print_distribution(
        "Train",
        train_records,
    )

    print_distribution(
        "Validation",
        val_records,
    )

    print_distribution(
        "Test",
        test_records,
    )

    # -----------------------------------------------------
    # StandardScaler
    # -----------------------------------------------------

    numeric_scaler = (
        StandardScaler()
    )

    # -----------------------------------------------------
    # Create loaders
    #
    # IMPORTANT:
    # fit_scaler=True ONLY for training.
    # -----------------------------------------------------

    train_dataset, train_loader = (
        create_loader(
            records=train_records,
            tokenizer=tokenizer,
            scaler=numeric_scaler,
            fit_scaler=True,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    val_dataset, val_loader = (
        create_loader(
            records=val_records,
            tokenizer=tokenizer,
            scaler=numeric_scaler,
            fit_scaler=False,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    test_dataset, test_loader = (
        create_loader(
            records=test_records,
            tokenizer=tokenizer,
            scaler=numeric_scaler,
            fit_scaler=False,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    print()

    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = HybridTransformerBugClassifier(

        vocab_size=VOCAB_SIZE,

        max_length=MAX_LENGTH,

        embedding_dim=EMBEDDING_DIM,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS,

        feedforward_dim=FEEDFORWARD_DIM,

        dropout=DROPOUT,

        numeric_feature_count=NUMERIC_FEATURE_COUNT,

        num_classes=NUM_CLASSES,

        pad_token_id=vocab["<PAD>"],
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()

    print(
        f"Total parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameter_count:,}"
    )

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    label_counts = np.bincount(
        [
            int(record["label"])
            for record in train_records
        ],
        minlength=NUM_CLASSES,
    )

    total_examples = (
        label_counts.sum()
    )

    class_weights = (
        total_examples
        /
        (
            NUM_CLASSES
            *
            np.maximum(
                label_counts,
                1
            )
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print()

    print(
        "Class counts:"
    )

    print(
        f"  label 0: "
        f"{label_counts[0]}"
    )

    print(
        f"  label 1: "
        f"{label_counts[1]}"
    )

    print(
        "Class weights:"
    )

    print(
        class_weights.detach()
        .cpu()
        .numpy()
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Mixed precision
    # -----------------------------------------------------

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            device.type == "cuda"
        )
    )

    # -----------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_val_f1 = -1.0

    best_val_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    print()

    print(
        "Starting training..."
    )

    print(
        "===================="
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()

        print(
            f"Epoch "
            f"{epoch}/{EPOCHS}"
        )

        print(
            "-------------------------"
        )

        train_loss, train_metrics = (
            train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
            )
        )

        print_metrics(
            "Training",
            train_loss,
            train_metrics,
        )

        (
            val_loss,
            val_metrics,
            _,
            _,
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print_metrics(
            "Validation",
            val_loss,
            val_metrics,
        )

        current_val_f1 = (
            val_metrics["f1"]
        )

        improved = False

        if current_val_f1 > best_val_f1:

            improved = True

        elif (
            current_val_f1 == best_val_f1
            and val_loss < best_val_loss
        ):

            improved = True

        if improved:

            best_val_f1 = (
                current_val_f1
            )

            best_val_loss = (
                val_loss
            )

            epochs_without_improvement = 0

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "best_val_f1":
                    best_val_f1,

                "best_val_loss":
                    best_val_loss,

                "config": {

                    "max_length":
                        MAX_LENGTH,

                    "vocab_size":
                        VOCAB_SIZE,

                    "embedding_dim":
                        EMBEDDING_DIM,

                    "num_heads":
                        NUM_HEADS,

                    "num_layers":
                        NUM_LAYERS,

                    "feedforward_dim":
                        FEEDFORWARD_DIM,

                    "dropout":
                        DROPOUT,

                    "numeric_feature_count":
                        NUMERIC_FEATURE_COUNT,

                    "num_classes":
                        NUM_CLASSES,
                },

                "scaler_mean":
                    numeric_scaler.mean_.tolist(),

                "scaler_scale":
                    numeric_scaler.scale_.tolist(),

            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

            print()

            print(
                "✓ Best checkpoint saved:"
            )

            print(
                f"  {CHECKPOINT_PATH}"
            )

        else:

            epochs_without_improvement += 1

            print()

            print(
                "No validation improvement."
            )

            print(
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "Early stopping."
            )

            break

    # -----------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------

    print()

    print(
        "Loading best checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation F1: "
        f"{checkpoint['best_val_f1']:.4f}"
    )

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    (
        val_loss,
        val_metrics,
        val_labels,
        val_predictions,
    ) = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print_metrics(
        "FINAL VALIDATION",
        val_loss,
        val_metrics,
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    (
        test_loss,
        test_metrics,
        test_labels,
        test_predictions,
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print_metrics(
        "FINAL TEST",
        test_loss,
        test_metrics,
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print()

    print(
        "Test Confusion Matrix"
    )

    print(
        "---------------------"
    )

    cm = confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1],
    )

    print(
        cm
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print()

    print(
        "Test Classification Report"
    )

    print(
        "--------------------------"
    )

    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=[0, 1],
            target_names=[
                "non_bug_fix",
                "bug_fix",
            ],
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print()

    print(
        "Training completed successfully."
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
