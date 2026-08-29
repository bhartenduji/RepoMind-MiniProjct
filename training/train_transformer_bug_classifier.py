import json
from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from training.bug_dataset import (
    BugPatchDataset,
)


from models.transformer_bug_classifier import (
    TransformerBugClassifier,
)


# =========================================================
# PATHS
# =========================================================

TRAIN_PATH = (
    "data/processed/"
    "classification_splits/train.jsonl"
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
    /
    "transformer_bug_classifier_v3.pt"
)


# =========================================================
# TRAINING CONFIG
# =========================================================

MAX_LENGTH = 256

BATCH_SIZE = 8

EPOCHS = 8

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.02

GRAD_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 2


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
# DATA
# =========================================================

def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def create_loader(
    records,
    tokenizer,
    batch_size,
    shuffle,
):

    dataset = BugPatchDataset(
        records,
        tokenizer,
        max_length=MAX_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        dataset,
        loader,
    )


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

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
):

    model.train()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

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

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
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
        len(loader)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
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

    all_labels = []
    all_predictions = []

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
            )

            loss = criterion(
                logits,
                labels,
            )

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    average_loss = (
        total_loss
        /
        len(loader)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return (
        average_loss,
        metrics,
        all_labels,
        all_predictions,
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
    print(name)
    print(
        "-------------------------"
    )

    print(
        f"Loss:      {loss:.4f}"
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
# MAIN
# =========================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "RepoMind Transformer Bug Classifier"
    )

    print(
        "===================================="
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
    # TOKENIZER
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
            f"Tokenizer vocabulary "
            f"({tokenizer_vocab_size}) "
            f"does not match "
            f"VOCAB_SIZE ({VOCAB_SIZE})."
        )

    vocab = tokenizer.get_vocab()

    pad_token_id = vocab.get(
        "<PAD>"
    )

    if pad_token_id is None:

        raise ValueError(
            "Tokenizer does not contain "
            "<PAD> token."
        )

    print(
        f"PAD token ID: "
        f"{pad_token_id}"
    )

    # -----------------------------------------------------
    # LOAD DATA
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

    print(
        f"Training records: "
        f"{len(train_records)}"
    )

    print(
        f"Validation records: "
        f"{len(val_records)}"
    )

    print(
        f"Test records: "
        f"{len(test_records)}"
    )

    # -----------------------------------------------------
    # CLASS DISTRIBUTION
    # -----------------------------------------------------

    def class_counts(records):

        counts = {
            0: 0,
            1: 0,
        }

        for record in records:

            label = int(
                record["label"]
            )

            if label not in counts:

                counts[label] = 0

            counts[label] += 1

        return counts

    train_counts = class_counts(
        train_records
    )

    val_counts = class_counts(
        val_records
    )

    test_counts = class_counts(
        test_records
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        f"Train: "
        f"0={train_counts.get(0, 0)}, "
        f"1={train_counts.get(1, 0)}"
    )

    print(
        f"Validation: "
        f"0={val_counts.get(0, 0)}, "
        f"1={val_counts.get(1, 0)}"
    )

    print(
        f"Test: "
        f"0={test_counts.get(0, 0)}, "
        f"1={test_counts.get(1, 0)}"
    )

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------

    print()
    print(
        "Creating DataLoaders..."
    )

    train_dataset, train_loader = (
        create_loader(
            train_records,
            tokenizer,
            BATCH_SIZE,
            shuffle=True,
        )
    )

    val_dataset, val_loader = (
        create_loader(
            val_records,
            tokenizer,
            BATCH_SIZE,
            shuffle=False,
        )
    )

    test_dataset, test_loader = (
        create_loader(
            test_records,
            tokenizer,
            BATCH_SIZE,
            shuffle=False,
        )
    )

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
    # MODEL
    # -----------------------------------------------------

    print()
    print(
        "Creating Transformer model..."
    )

    model = TransformerBugClassifier(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        pad_token_id=pad_token_id,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    # -----------------------------------------------------
    # CLASS WEIGHT
    # -----------------------------------------------------

    positive_count = train_counts.get(
        1,
        0,
    )

    negative_count = train_counts.get(
        0,
        0,
    )

    if positive_count == 0 or negative_count == 0:

        raise ValueError(
            "Training dataset must contain "
            "both class 0 and class 1."
        )

    # Weight minority class.
    class_weight = torch.tensor(
        [
            1.0,
            negative_count / positive_count,
        ],
        dtype=torch.float32,
        device=device,
    )

    print()
    print(
        "Class weights:"
    )

    print(
        f"Class 0: "
        f"{class_weight[0].item():.4f}"
    )

    print(
        f"Class 1: "
        f"{class_weight[1].item():.4f}"
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weight
    )

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            device.type == "cuda"
        )
    )

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_f1 = -1.0

    patience_counter = 0

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
            "=" * 60
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "=" * 60
        )

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        )

        val_loss, val_metrics, _, _ = (
            evaluate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        print_metrics(
            "Training",
            train_loss,
            train_metrics,
        )

        print_metrics(
            "Validation",
            val_loss,
            val_metrics,
        )

        # -------------------------------------------------
        # CHECKPOINT
        # -------------------------------------------------

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = (
                val_metrics["f1"]
            )

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_val_f1":
                        best_val_f1,

                    "vocab_size":
                        VOCAB_SIZE,

                    "max_length":
                        MAX_LENGTH,

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

                    "pad_token_id":
                        pad_token_id,
                },
                CHECKPOINT_PATH,
            )

            print()
            print(
                "✓ Best model saved:"
            )

            print(
                CHECKPOINT_PATH
            )

        else:

            patience_counter += 1

            print()
            print(
                f"No validation F1 improvement."
            )

            print(
                f"Patience: "
                f"{patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print()
                print(
                    "Early stopping."
                )

                break

    # -----------------------------------------------------
    # LOAD BEST MODEL
    # -----------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Best checkpoint was not created."
        )

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
        f"Best validation F1: "
        f"{checkpoint['best_val_f1']:.4f}"
    )

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    test_loss, test_metrics, labels, predictions = (
        evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
    )

    print_metrics(
        "FINAL TEST",
        test_loss,
        test_metrics,
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    print()
    print(
        "Confusion Matrix"
    )

    print(
        "================"
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    print(cm)

    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    print()
    print(
        "Classification Report"
    )

    print(
        "====================="
    )

    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[
                "non_bug",
                "bug_fix",
            ],
            zero_division=0,
        )
    )

    print()
    print(
        "=" * 60
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()