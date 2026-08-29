import json
from pathlib import Path

from inference.predict_bug_fix import load_model, predict

TEST = Path("data/processed/classification_splits/test.jsonl")
OUT = Path("data/processed/classification_splits/test_predictions.jsonl")

model, tokenizer, checkpoint = load_model()

records = [json.loads(x) for x in TEST.open()]

correct = 0

with OUT.open("w") as f:
    for i, record in enumerate(records):

        result = predict(
            record,
            model,
            tokenizer,
            checkpoint,
        )

        actual = int(record["label"])
        predicted = 1 if result["prediction"] == "bug_fix" else 0

        correct += predicted == actual

        f.write(json.dumps({
            "index": i,
            "actual": actual,
            "prediction": result["prediction"],
            "bug_fix_probability": result["bug_fix_probability"],
            "non_bug_fix_probability": result["non_bug_fix_probability"],
        }) + "\n")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(records)}")

print()
print("Records:", len(records))
print("Correct:", correct)
print("Accuracy:", f"{correct / len(records):.4f}")
print("Saved:", OUT)
