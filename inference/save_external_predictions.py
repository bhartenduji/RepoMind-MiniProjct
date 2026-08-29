import json
from pathlib import Path
import numpy as np
import torch

from inference.predict_bug_fix import (
    load_model, encode_patch, get_numeric_features
)

DATA = Path("data/processed/classification_splits/bug_fix_eval_dataset_large.jsonl")
OUT = Path("data/processed/classification_splits/external_predictions.jsonl")

model, tokenizer, checkpoint = load_model()
config = checkpoint["config"]
vocab = tokenizer.get_vocab()

mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)

records = [json.loads(x) for x in DATA.open() if x.strip()]

with OUT.open("w") as f:
    with torch.no_grad():
        for i, r in enumerate(records):
            ids, mask = encode_patch(
                tokenizer, r.get("patch", ""),
                vocab["<BOS>"], vocab["<EOS>"], vocab["<PAD>"],
                config["max_length"]
            )

            numeric = np.asarray(
                get_numeric_features(r), dtype=np.float32
            )
            numeric = (numeric - mean) / scale

            logits = model(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.tensor([mask]),
                numeric_features=torch.tensor([numeric])
            )

            probs = torch.softmax(logits, dim=1)[0]

            f.write(json.dumps({
                "index": i,
                "actual": int(r["label"]),
                "bug_fix_probability": float(probs[1]),
                "non_bug_fix_probability": float(probs[0])
            }) + "\n")

            if (i + 1) % 25 == 0:
                print(f"Processed {i+1}/{len(records)}")

print("Saved:", OUT)
