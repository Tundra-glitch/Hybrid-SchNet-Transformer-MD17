"""
transformer.py

Custom Multi-Head Transformer module
for molecular graph representation learning.

Project:
Hybrid SchNet + Multi-Head Transformer
"""


from typing import Optional


import torch

from torch import Tensor

import torch.nn as nn

import torch.nn.functional as F



# ============================================================
# Multi Head Attention
# ============================================================


class MultiHeadAttention(nn.Module):
    """
    Custom multi-head self attention.

    Parameters
    ----------
    hidden_dim:
        Feature dimension.

    num_heads:
        Number of attention heads.

    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):

        super().__init__()


        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads"
            )


        self.hidden_dim = hidden_dim

        self.num_heads = num_heads


        self.head_dim = (
            hidden_dim // num_heads
        )


        self.query = nn.Linear(
            hidden_dim,
            hidden_dim,
        )


        self.key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )


        self.value = nn.Linear(
            hidden_dim,
            hidden_dim,
        )


        self.output = nn.Linear(
            hidden_dim,
            hidden_dim,
        )


        self.dropout = nn.Dropout(
            dropout
        )



    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Atom features.

            Shape:
            [batch, atoms, hidden_dim]


        Returns
        -------
        Tensor
            Updated atom features.
        """


        batch_size, num_atoms, _ = x.shape



        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)



        # Split into multiple heads

        Q = Q.view(
            batch_size,
            num_atoms,
            self.num_heads,
            self.head_dim,
        )


        K = K.view(
            batch_size,
            num_atoms,
            self.num_heads,
            self.head_dim,
        )


        V = V.view(
            batch_size,
            num_atoms,
            self.num_heads,
            self.head_dim,
        )



        # Move heads forward

        Q = Q.transpose(1,2)

        K = K.transpose(1,2)

        V = V.transpose(1,2)



        # Attention score

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )


        scores = scores / (
            self.head_dim ** 0.5
        )



        if mask is not None:

            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )



        attention = F.softmax(
            scores,
            dim=-1,
        )


        attention = self.dropout(
            attention
        )


        output = torch.matmul(
            attention,
            V,
        )



        # Combine heads

        output = output.transpose(
            1,2
        )


        output = output.contiguous()


        output = output.view(
            batch_size,
            num_atoms,
            self.hidden_dim,
        )


        output = self.output(
            output
        )


        return output




# ============================================================
# Transformer Encoder Block
# ============================================================


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.
    """


    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        expansion: int = 4,
    ):

        super().__init__()



        self.attention = MultiHeadAttention(
            hidden_dim,
            num_heads,
            dropout,
        )



        self.norm1 = nn.LayerNorm(
            hidden_dim
        )



        self.feed_forward = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * expansion,
            ),

            nn.SiLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim * expansion,
                hidden_dim,
            ),
        )



        self.norm2 = nn.LayerNorm(
            hidden_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )




    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Transformer forward pass.
        """


        attention_output = (
            self.attention(x)
        )


        x = x + self.dropout(
            attention_output
        )


        x = self.norm1(
            x
        )



        ff_output = (
            self.feed_forward(x)
        )


        x = x + self.dropout(
            ff_output
        )


        x = self.norm2(
            x
        )


        return x




# ============================================================
# Transformer Encoder
# ============================================================


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks.
    """


    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):

        super().__init__()


        self.layers = nn.ModuleList(

            [

                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    dropout,
                )

                for _ in range(num_layers)

            ]

        )



    def forward(
        self,
        x: Tensor,
    ) -> Tensor:


        for layer in self.layers:

            x = layer(x)


        return x



# ============================================================
# Test
# ============================================================


if __name__ == "__main__":


    model = TransformerEncoder(
        hidden_dim=128,
        num_heads=8,
        num_layers=2,
    )


    fake_atoms = torch.randn(
        4,
        20,
        128,
    )


    output = model(
        fake_atoms
    )


    print(
        "Input:",
        fake_atoms.shape
    )


    print(
        "Output:",
        output.shape
    )
