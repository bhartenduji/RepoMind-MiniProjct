import json
from pathlib import Path

import numpy as np
import torch

from inference.predict_bug_fix import (
    load_model,
    encode_patch,
    get_numeric_features,
)

DATA = Path(
    "data/processed/classification_splits/"
    "bug_fix_eval_dataset_large.jsonl"
)

THRESHOLD = 0.31


def main():
    model, tokenizer, checkpoint = load_model()

    config = checkpoint["config"]
    vocab = tokenizer.get_vocab()

    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]
    pad_id = vocab["<PAD>"]

    mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float32,
    )

    records = [
        json.loads(line)
        for line in DATA.open()
        if line.strip()
    ]

    tp = fp = tn = fn = 0

    print("External records:", len(records))
    print("Threshold:", THRESHOLD)
    print("Running inference...")

    with torch.no_grad():

        for i, record in enumerate(records):

            ids, mask = encode_patch(
                tokenizer,
                record.get("patch", ""),
                bos_id,
                eos_id,
                pad_id,
                config["max_length"],
            )

            numeric = np.asarray(
                get_numeric_features(record),
                dtype=np.float32,
            )

            numeric = (numeric - mean) / scale

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

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_features=numeric_features,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

            probability = float(
                probabilities[1].item()
            )

            predicted = (
                1
                if probability >= THRESHOLD
                else 0
            )

            actual = int(record["label"])

            if actual == 1 and predicted == 1:
                tp += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 0 and predicted == 0:
                tn += 1
            else:
                fn += 1

            if (i + 1) % 25 == 0:
                print(
                    f"Processed {i + 1}/{len(records)}"
                )

    accuracy = (tp + tn) / len(records)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    print()
    print("=" * 45)
    print("EXTERNAL / UNSEEN DATASET")
    print("=" * 45)
    print("Records:", len(records))
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print()
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)
    print()
    print("Confusion Matrix:")
    print(f"[[{tn:4d} {fp:4d}]")
    print(f" [{fn:4d} {tp:4d}]]")


if __name__ == "__main__":
    main()
