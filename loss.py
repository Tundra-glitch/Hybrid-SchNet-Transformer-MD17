"""
loss.py

Loss functions and metrics for molecular
energy and force prediction.

The total loss combines:

Energy loss

+

Force loss


L = lambda_E * L_E
    +
    lambda_F * L_F


Project:
Hybrid-SchNet-Transformer-MD17
"""


from typing import Dict


import torch

from torch import Tensor

import torch.nn as nn


from config import config



# ============================================================
# Energy Loss
# ============================================================


class EnergyLoss(nn.Module):
    """
    Mean squared error loss for energy prediction.
    """


    def __init__(self):

        super().__init__()


        self.loss = nn.MSELoss()



    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:


        return self.loss(
            prediction,
            target,
        )





# ============================================================
# Force Loss
# ============================================================


class ForceLoss(nn.Module):
    """
    Mean squared error loss for atomic forces.
    """


    def __init__(self):

        super().__init__()


        self.loss = nn.MSELoss()



    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:


        return self.loss(

            prediction,

            target,

        )





# ============================================================
# Combined Molecular Loss
# ============================================================


class MolecularLoss(nn.Module):
    """
    Combined energy + force loss.

    L =
    energy_weight * energy_loss
    +
    force_weight * force_loss
    """


    def __init__(self):

        super().__init__()


        self.energy_loss = EnergyLoss()

        self.force_loss = ForceLoss()



    def forward(
        self,
        predicted_energy: Tensor,
        true_energy: Tensor,
        predicted_forces: Tensor,
        true_forces: Tensor,

    ) -> Dict[str, Tensor]:


        energy_error = self.energy_loss(

            predicted_energy,

            true_energy,

        )



        force_error = self.force_loss(

            predicted_forces,

            true_forces,

        )



        total_loss = (

            config.energy_loss_weight
            *
            energy_error

            +

            config.force_loss_weight
            *
            force_error

        )



        return {

            "loss": total_loss,

            "energy_loss": energy_error,

            "force_loss": force_error,

        }





# ============================================================
# Metrics
# ============================================================


def mean_absolute_error(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Calculate MAE.
    """


    return torch.mean(

        torch.abs(

            prediction-target

        )

    )





def calculate_metrics(
    energy_prediction: Tensor,
    energy_target: Tensor,
    force_prediction: Tensor,
    force_target: Tensor,

) -> Dict[str,float]:
    """
    Calculate evaluation metrics.

    Returns:
        Energy MAE
        Force MAE
    """


    energy_mae = mean_absolute_error(

        energy_prediction,

        energy_target,

    )


    force_mae = mean_absolute_error(

        force_prediction,

        force_target,

    )



    return {


        "energy_MAE":
            energy_mae.item(),


        "force_MAE":
            force_mae.item(),


    }





# ============================================================
# Test
# ============================================================


if __name__ == "__main__":


    predicted_energy = torch.tensor(

        [[1.2]]

    )


    true_energy = torch.tensor(

        [[1.0]]

    )



    predicted_force = torch.randn(

        5,

        3,

    )


    true_force = torch.randn(

        5,

        3,

    )



    criterion = MolecularLoss()



    output = criterion(

        predicted_energy,

        true_energy,

        predicted_force,

        true_force,

    )


    print(output)
