"""
model.py

Hybrid SchNet + Multi-Head Transformer
for molecular energy and force prediction.

Architecture:

Atomic structure
        |
        v
SchNet-style atom encoder
        |
        v
Transformer attention
        |
        v
Energy prediction
        |
        v
Force calculation

F = -dE/dR

Project:
Hybrid-SchNet-Transformer-MD17
"""


from typing import Tuple


import torch

from torch import Tensor

import torch.nn as nn


from torch_geometric.nn import (
    MessagePassing,
    radius_graph,
    global_add_pool,
)


from torch_geometric.nn.inits import glorot


from transformer import TransformerEncoder

from config import config



# ============================================================
# Continuous Filter Layer
# ============================================================


class RadialFilter(nn.Module):
    """
    Learn distance-dependent interaction filters.

    This is the key idea behind SchNet:
    molecular interactions depend on interatomic distances.
    """


    def __init__(
        self,
        hidden_dim: int,
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(1, hidden_dim),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            )

        )



    def forward(
        self,
        distance: Tensor,
    ) -> Tensor:


        distance = distance.unsqueeze(-1)


        return self.network(distance)




# ============================================================
# SchNet Interaction Block
# ============================================================


class SchNetInteraction(MessagePassing):
    """
    Simplified SchNet interaction block.

    Atom features exchange information
    through distance-based messages.
    """


    def __init__(
        self,
        hidden_dim: int,
    ):

        super().__init__(
            aggr="add"
        )


        self.filter = RadialFilter(
            hidden_dim
        )


        self.message_network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            )

        )



        self.update = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.SiLU(),

        )




    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:


        row, col = edge_index


        distance = torch.norm(
            pos[row]-pos[col],
            dim=1,
        )


        return self.propagate(

            edge_index,

            x=x,

            distance=distance,

        )




    def message(
        self,
        x_j: Tensor,
        distance: Tensor,
    ) -> Tensor:


        filter_weight = self.filter(
            distance
        )


        return x_j * filter_weight



    def update(
        self,
        aggr_out: Tensor,
    ) -> Tensor:


        return self.update(
            aggr_out
        )




# ============================================================
# SchNet Encoder
# ============================================================


class SchNetEncoder(nn.Module):
    """
    Atom-level SchNet encoder.

    Output:
    one feature vector per atom.
    """


    def __init__(
        self,
        hidden_dim: int,
        num_interactions: int,
    ):

        super().__init__()


        self.embedding = nn.Embedding(
            100,
            hidden_dim,
        )


        self.interactions = nn.ModuleList(

            [

                SchNetInteraction(
                    hidden_dim
                )

                for _ in range(num_interactions)

            ]

        )



    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:


        x = self.embedding(
            z
        )


        for interaction in self.interactions:


            x = x + interaction(

                x,

                pos,

                edge_index,

            )


        return x




# ============================================================
# Energy Head
# ============================================================


class EnergyHead(nn.Module):


    def __init__(
        self,
        hidden_dim:int,
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
                1,
            )

        )



    def forward(
        self,
        x: Tensor,
    ) -> Tensor:


        return self.network(x)




# ============================================================
# Hybrid Model
# ============================================================


class HybridSchNetTransformer(nn.Module):
    """
    Final hybrid molecular model.
    """


    def __init__(self):

        super().__init__()



        self.encoder = SchNetEncoder(

            config.hidden_dim,

            config.num_interactions,

        )



        self.transformer = TransformerEncoder(

            hidden_dim=config.hidden_dim,

            num_heads=config.num_attention_heads,

            num_layers=config.num_transformer_layers,

            dropout=config.dropout,

        )



        self.energy_head = EnergyHead(

            config.hidden_dim

        )




    def forward(
        self,
        data,
        return_forces=True,
    ) -> Tuple[Tensor, Tensor]:


        z = data.z

        pos = data.pos

        batch = data.batch



        if return_forces:

            pos.requires_grad_(True)



        # build molecular graph

        edge_index = radius_graph(

            pos,

            r=5.0,

            batch=batch,

        )



        # SchNet atom features

        atom_features = self.encoder(

            z,

            pos,

            edge_index,

        )



        # Transformer needs:

        # [batch, atoms, features]

        atom_features = atom_features.unsqueeze(0)



        atom_features = self.transformer(

            atom_features

        )



        atom_features = atom_features.squeeze(0)



        # molecular representation

        molecule_features = global_add_pool(

            atom_features,

            batch,

        )



        energy = self.energy_head(

            molecule_features

        )



        forces = None


        if return_forces:


            forces = -torch.autograd.grad(

                energy.sum(),

                pos,

                create_graph=True,

            )[0]



        return energy, forces




# ============================================================
# Test
# ============================================================


if __name__ == "__main__":


    model = HybridSchNetTransformer()


    print(model)
