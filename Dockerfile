FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY api ./api
COPY inference ./inference
COPY models ./models
COPY data/processed/code_tokenizer ./data/processed/code_tokenizer
COPY checkpoints/hybrid_transformer_bug_classifier_augmented_500.pt ./checkpoints/hybrid_transformer_bug_classifier_augmented_500.pt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
