import torch
import torch.nn as nn


class HybridTransformerBugClassifier(nn.Module):

    def __init__(
        self,
        vocab_size=16000,
        max_length=1024,
        embedding_dim=128,
        num_heads=4,
        num_layers=3,
        feedforward_dim=256,
        dropout=0.25,
        numeric_feature_count=15,
        num_classes=2,
        pad_token_id=0,
    ):

        super().__init__()

        self.max_length = max_length

        # =====================================================
        # TOKEN EMBEDDINGS
        # =====================================================

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim,
        )

        # =====================================================
        # TRANSFORMER ENCODER
        # =====================================================

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
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # =====================================================
        # NUMERIC METADATA NETWORK
        # =====================================================

        # =====================================================
# NUMERIC METADATA NETWORK
# =====================================================

        self.metadata_network = nn.Sequential(

            nn.Linear(
                numeric_feature_count,
                    64,
                ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
                    ),

            nn.Linear(
                64,
                32,
                ),

            nn.ReLU(),

            )
        # =====================================================
        # FINAL CLASSIFIER
        # =====================================================

        combined_size = (
            embedding_dim
            + 32
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                combined_size,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                128,
                num_classes,
            ),

        )

    # =========================================================
    # MASKED MEAN POOLING
    # =========================================================

    def masked_mean_pool(
        self,
        hidden_states,
        attention_mask,
    ):

        mask = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )

        hidden_states = (
            hidden_states
            * mask
        )

        summed = hidden_states.sum(
            dim=1
        )

        count = mask.sum(
            dim=1
        ).clamp(
            min=1e-6
        )

        return (
            summed
            / count
        )

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(
        self,
        input_ids,
        attention_mask,
        numeric_features,
    ):

        batch_size, sequence_length = (
            input_ids.shape
        )

        # =====================================================
        # LENGTH CHECK
        # =====================================================

        if sequence_length > self.max_length:

            raise ValueError(
                f"Sequence length "
                f"{sequence_length} exceeds "
                f"model max_length "
                f"{self.max_length}."
            )

        # =====================================================
        # POSITIONS
        # =====================================================

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        positions = (
            positions
            .unsqueeze(0)
            .expand(
                batch_size,
                sequence_length,
            )
        )

        # =====================================================
        # TOKEN EMBEDDINGS
        # =====================================================

        token_embeddings = (
            self.token_embedding(
                input_ids
            )
        )

        # =====================================================
        # POSITION EMBEDDINGS
        # =====================================================

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        # =====================================================
        # COMBINE TOKEN + POSITION
        # =====================================================

        hidden_states = (
            token_embeddings
            + position_embeddings
        )

        # =====================================================
        # PADDING MASK
        # =====================================================

        padding_mask = (
            attention_mask == 0
        )

        # =====================================================
        # TRANSFORMER
        # =====================================================

        hidden_states = self.encoder(
            hidden_states,
            src_key_padding_mask=padding_mask,
        )

        # =====================================================
        # PATCH REPRESENTATION
        # =====================================================

        patch_embedding = (
            self.masked_mean_pool(
                hidden_states,
                attention_mask,
            )
        )

        # =====================================================
        # NUMERIC REPRESENTATION
        # =====================================================

        metadata_embedding = (
            self.metadata_network(
                numeric_features
            )
        )

        # =====================================================
        # COMBINE PATCH + METADATA
        # =====================================================

        combined = torch.cat(
            [
                patch_embedding,
                metadata_embedding,
            ],
            dim=1,
        )

        # =====================================================
        # CLASSIFICATION
        # =====================================================

        logits = self.classifier(
            combined
        )

        return logits