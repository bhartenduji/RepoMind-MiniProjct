# RepoMind

RepoMind is a repository-aware bug-fix classification project using code diffs, repository history, and a hybrid Transformer-based classifier.

## API

RepoMind provides a FastAPI endpoint for predicting whether a code patch represents a bug fix.

### Run with Docker

Build the image:

```bash
docker build -t repomind-api .
