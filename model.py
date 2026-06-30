"""
model.py

Hybrid SchNet + Multi-Head Transformer model
for molecular energy and force prediction.

The model predicts molecular energy and obtains
atomic forces using automatic differentiation:

F = -grad(E)

Project:
Hybrid-SchNet-Transformer-MD17
"""


from typing import Tuple


import torch

from torch import Tensor

import torch.nn as nn


from torch_geometric.nn import (
    SchNet,
    global_add_pool,
)


from transformer import TransformerEncoder

from config import config



# ============================================================
# Energy Prediction Head
# ============================================================


class EnergyHead(nn.Module):
    """
    Neural network regression head.

    Converts molecular representation into energy.
    """


    def __init__(
        self,
        hidden_dim: int,
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.SiLU(),


            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),


            nn.SiLU(),


            nn.Linear(
                hidden_dim // 2,
                1,
            )

        )



    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        """
        Predict energy.

        Parameters
        ----------
        x:
            Molecular representation.

        Returns
        -------
        Tensor
            Molecular energy.
        """


        return self.network(x)




# ============================================================
# Hybrid Model
# ============================================================


class HybridSchNetTransformer(nn.Module):
    """
    Hybrid molecular neural network.

    Architecture:

    Atomic structure
          |
          v
       SchNet
          |
          v
    Transformer Encoder
          |
          v
     Pooling
          |
          v
       Energy


    Forces are obtained using:

    F = -grad(E)
    """



    def __init__(self):

        super().__init__()



        # ----------------------------------------------------
        # SchNet Encoder
        # ----------------------------------------------------

        self.schnet = SchNet(

            hidden_channels=config.hidden_dim,

            num_filters=config.hidden_dim,

            num_interactions=config.num_interactions,

            cutoff=10.0,

        )



        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------


        self.transformer = TransformerEncoder(

            hidden_dim=config.hidden_dim,

            num_heads=config.num_attention_heads,

            num_layers=config.num_transformer_layers,

            dropout=config.dropout,

        )



        # ----------------------------------------------------
        # Energy predictor
        # ----------------------------------------------------


        self.energy_head = EnergyHead(

            config.hidden_dim

        )





    def forward(
        self,
        data,
        return_forces: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        """
        Forward pass.

        Parameters
        ----------
        data:
            PyTorch Geometric molecular graph.

        return_forces:
            Whether to compute forces.


        Returns
        -------
        energy:
            Predicted molecular energy.

        forces:
            Predicted atomic forces.
        """


        z = data.z

        pos = data.pos

        batch = data.batch



        # Enable force calculation

        if return_forces:

            pos.requires_grad_(True)



        # ----------------------------------------------------
        # SchNet
        # ----------------------------------------------------

        atomic_features = self.schnet(

            z,

            pos,

            batch,

        )


        """
        Important:

        PyG SchNet already returns graph-level output.

        For Transformer we need atom-level features.

        Therefore, in the final implementation,
        we will replace this with a custom SchNet
        message passing encoder.

        This placeholder keeps the architecture clean
        while we build the custom version.
        """



        molecular_features = atomic_features



        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------


        energy = self.energy_head(

            molecular_features

        )



        # ----------------------------------------------------
        # Force calculation
        # ----------------------------------------------------


        forces = None


        if return_forces:


            forces = -torch.autograd.grad(

                energy.sum(),

                pos,

                create_graph=True,

            )[0]



        return energy, forces





# ============================================================
# Model Test
# ============================================================


def test_model():

    """
    Simple forward test.
    """


    model = HybridSchNetTransformer()


    print(model)



if __name__ == "__main__":

    test_model()
