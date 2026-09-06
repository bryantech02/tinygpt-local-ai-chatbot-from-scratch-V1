# ============================================================
# TINYGPT
# PHASE 18 — COMPLETE TRANSFORMER ARCHITECTURE BY BRYAN TORCULAS
# UPDATED FOR RAG TRAINING
# ============================================================

import math

import torch
import torch.nn as nn

from tokenizer import VOCAB_SIZE




# ============================================================
# 1. CONFIGURATION
# ============================================================

VOCAB_SIZE = 132

EMBEDDING_SIZE = 64

MAX_SEQUENCE_LENGTH = 512

NUMBER_OF_HEADS = 4

NUMBER_OF_LAYERS = 3

FEED_FORWARD_SIZE = 256

DROPOUT = 0.1


# ============================================================
# 2. TOKEN EMBEDDING
# ============================================================

class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )


    def forward(self, tokens):

        return self.embedding(tokens)


# ============================================================
# 3. POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embedding_size,
        max_sequence_length
    ):

        super().__init__()

        position = torch.arange(
            max_sequence_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_size,
                2
            )
            * (
                -math.log(10000.0)
                / embedding_size
            )
        )

        positional_encoding = torch.zeros(
            max_sequence_length,
            embedding_size
        )

        positional_encoding[
            :,
            0::2
        ] = torch.sin(
            position * div_term
        )

        positional_encoding[
            :,
            1::2
        ] = torch.cos(
            position * div_term
        )

        positional_encoding = (
            positional_encoding.unsqueeze(0)
        )

        self.register_buffer(
            "positional_encoding",
            positional_encoding
        )


    def forward(self, x):

        sequence_length = x.size(1)

        if sequence_length > self.positional_encoding.size(1):

            raise ValueError(
                "Input sequence length "
                f"({sequence_length}) exceeds "
                "the positional encoding limit "
                f"({self.positional_encoding.size(1)})."
            )

        return (
            x
            + self.positional_encoding[
                :,
                :sequence_length,
                :
            ]
        )


# ============================================================
# 4. MULTI-HEAD SELF-ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_size,
        number_of_heads,
        dropout
    ):

        super().__init__()

        if embedding_size % number_of_heads != 0:

            raise ValueError(
                "Embedding size must be divisible "
                "by number of attention heads."
            )


        self.embedding_size = embedding_size

        self.number_of_heads = number_of_heads

        self.head_size = (
            embedding_size
            // number_of_heads
        )


        self.query = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.key = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.value = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        batch_size, sequence_length, embedding_size = x.shape


        # ----------------------------------------------------
        # Create Q, K, V
        # ----------------------------------------------------

        query = self.query(x)

        key = self.key(x)

        value = self.value(x)


        # ----------------------------------------------------
        # Split into attention heads
        # ----------------------------------------------------

        query = query.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_size
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_size
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_size
        ).transpose(1, 2)


        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        )

        scores = (
            scores
            / math.sqrt(self.head_size)
        )


        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()


        scores = scores.masked_fill(
            causal_mask,
            float("-inf")
        )


        # ----------------------------------------------------
        # Attention probabilities
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )


        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        output = torch.matmul(
            attention_weights,
            value
        )


        # ----------------------------------------------------
        # Combine heads
        # ----------------------------------------------------

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_size
        )


        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        output = self.output_projection(
            output
        )


        return output


# ============================================================
# 5. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_size,
        feed_forward_size,
        dropout
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_size,
                feed_forward_size
            ),

            nn.GELU(),

            nn.Linear(
                feed_forward_size,
                embedding_size
            ),

            nn.Dropout(
                dropout
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 6. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        number_of_heads,
        feed_forward_size,
        dropout
    ):

        super().__init__()


        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = MultiHeadAttention(
            embedding_size,
            number_of_heads,
            dropout
        )


        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = FeedForward(
            embedding_size,
            feed_forward_size,
            dropout
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Attention residual connection
        # ----------------------------------------------------

        x = x + self.attention(
            self.layer_norm_1(x)
        )


        # ----------------------------------------------------
        # Feed-forward residual connection
        # ----------------------------------------------------

        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )


        return x


# ============================================================
# 7. COMPLETE TINYGPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_layers=NUMBER_OF_LAYERS,
        feed_forward_size=FEED_FORWARD_SIZE,
        dropout=DROPOUT
    ):

        super().__init__()


        self.vocab_size = vocab_size

        self.embedding_size = embedding_size

        self.max_sequence_length = (
            max_sequence_length
        )


        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_size
        )


        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.position_encoding = PositionalEncoding(
            embedding_size,
            max_sequence_length
        )


        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.transformer_blocks = nn.ModuleList(

            [

                TransformerBlock(
                    embedding_size,
                    number_of_heads,
                    feed_forward_size,
                    dropout
                )

                for _ in range(number_of_layers)

            ]

        )


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )


        # ----------------------------------------------------
        # Language modeling head
        # ----------------------------------------------------

        self.language_head = nn.Linear(
            embedding_size,
            vocab_size
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, tokens):

        # ----------------------------------------------------
        # Check sequence length
        # ----------------------------------------------------

        if tokens.size(1) > self.max_sequence_length:

            raise ValueError(
                "Input sequence is longer than "
                f"the maximum sequence length of "
                f"{self.max_sequence_length}."
            )


        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        x = self.token_embedding(
            tokens
        )


        # ----------------------------------------------------
        # Add positional information
        # ----------------------------------------------------

        x = self.position_encoding(
            x
        )


        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        for block in self.transformer_blocks:

            x = block(x)


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_layer_norm(
            x
        )


        # ----------------------------------------------------
        # Predict vocabulary logits
        # ----------------------------------------------------

        logits = self.language_head(
            x
        )


        return logits


# ============================================================
# 8. CREATE TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = TinyGPT()


    # ========================================================
    # 9. DISPLAY MODEL
    # ========================================================

    print()

    print("======================================")

    print("     TINYGPT COMPLETE TRANSFORMER")

    print("======================================")

    print()

    print(model)


    # ========================================================
    # 10. COUNT PARAMETERS
    # ========================================================

    parameter_count = sum(

        parameter.numel()

        for parameter in model.parameters()

    )


    print()

    print(
        "Trainable parameters:",
        parameter_count
    )


    # ========================================================
    # 11. TEST INPUT
    # ========================================================

    test_tokens = torch.tensor(

        [
            [
                2,
                13,
                21,
                25,
                34,
                9,
                12,
                13
            ]
        ],

        dtype=torch.long

    )


    print()

    print("Test tokens:")

    print(test_tokens)


    print()

    print("Input shape:")

    print(test_tokens.shape)


    # ========================================================
    # 12. FORWARD PASS
    # ========================================================

    with torch.no_grad():

        output = model(
            test_tokens
        )


    print()

    print("Output shape:")

    print(output.shape)


    # ========================================================
    # 13. EXPECTED OUTPUT
    # ========================================================

    print()

    print("Expected output shape:")

    print(
        (
            1,
            8,
            VOCAB_SIZE
        )
    )


    # ========================================================
    # 14. TEST 512 TOKEN CAPABILITY
    # ========================================================

    print()

    print("Testing 512-token sequence...")


    long_test_tokens = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(1, 512),
        dtype=torch.long
    )


    with torch.no_grad():

        long_output = model(
            long_test_tokens
        )


    print()

    print("Long input shape:")

    print(long_test_tokens.shape)


    print()

    print("Long output shape:")

    print(long_output.shape)


    print()

    print("Expected long output shape:")

    print(
        (
            1,
            512,
            VOCAB_SIZE
        )
    )


    # ========================================================
    # 15. SUCCESS
    # ========================================================

    print()

    print("======================================")

    print("          SUCCESS!")

    print("======================================")

    print()

    print(
        "TinyGPT complete Transformer architecture "
        "is working."
    )

    print()

    print(
        "Maximum sequence length:"
    )

    print(
        MAX_SEQUENCE_LENGTH
    )

    print()

    print(
        "TinyGPT now contains:"
    )

    print()

    print("✓ Token embeddings")

    print("✓ Positional encoding")

    print("✓ Multi-head self-attention")

    print("✓ Causal masking")

    print("✓ Feed-forward networks")

    print("✓ Residual connections")

    print("✓ Layer normalization")

    print("✓ Multiple Transformer blocks")

    print("✓ Language modeling head")

    print("✓ 512-token sequence support")
