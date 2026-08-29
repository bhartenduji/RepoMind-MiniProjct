import torch
import torch.nn as nn


class TransformerBugClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=16000,
        max_length=256,
        embedding_dim=256,
        num_heads=8,
        num_layers=4,
        feedforward_dim=512,
        dropout=0.1,
        num_classes=2,
        pad_token_id=0,
    ):
        super().__init__()

        self.max_length = max_length
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def masked_mean_pool(
        self,
        hidden_states,
        attention_mask
    ):
        mask = attention_mask.unsqueeze(
            -1
        ).float()

        hidden_states = (
            hidden_states
            *
            mask
        )

        summed = hidden_states.sum(
            dim=1
        )

        counts = mask.sum(
            dim=1
        ).clamp(
            min=1e-9
        )

        return summed / counts

    def forward(
        self,
        input_ids,
        attention_mask
    ):
        batch_size, sequence_length = (
            input_ids.shape
        )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        positions = positions.unsqueeze(
            0
        ).expand(
            batch_size,
            sequence_length
        )

        token_embeddings = (
            self.token_embedding(
                input_ids
            )
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        hidden_states = (
            token_embeddings
            +
            position_embeddings
        )

        padding_mask = (
            attention_mask == 0
        )

        hidden_states = self.encoder(
            hidden_states,
            src_key_padding_mask=padding_mask
        )

        pooled = self.masked_mean_pool(
            hidden_states,
            attention_mask
        )

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits