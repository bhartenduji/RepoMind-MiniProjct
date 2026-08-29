# RepoMind

RepoMind is a repository-aware bug-fix classification project using code diffs, repository history, and a hybrid Transformer-based classifier.

## API

RepoMind provides a FastAPI endpoint for predicting whether a code patch represents a bug fix.

### Run with Docker

Build the image:

```bash
docker build -t repomind-api .


## Model Evaluation

The current model was evaluated on the held-out test split with 1,979 records.

| Metric | Score |
|---|---:|
| Accuracy | 77.92% |
| Precision | 81.19% |
| Recall | 86.37% |
| F1 | 83.70% |

Confusion matrix:

```text
[[420 260]
 [177 1122]]
```

Evaluation command:

```bash
python evaluate_test.py
```
