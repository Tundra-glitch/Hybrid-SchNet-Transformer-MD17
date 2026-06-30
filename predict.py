"""
predict.py

Inference script for Hybrid SchNet + Transformer.

Given a molecular structure:

- Atomic numbers
- 3D coordinates

predict:

- Molecular energy
- Atomic forces


Run:

python predict.py


Project:
Hybrid-SchNet-Transformer-MD17
"""


from pathlib import Path


import torch


from torch_geometric.data import Data


from model import HybridSchNetTransformer


from config import config




# ============================================================
# Load trained model
# ============================================================


def load_model():

    """
    Load best checkpoint.
    """


    model = HybridSchNetTransformer()



    checkpoint_path = (

        Path(config.checkpoint_dir)

        /

        "best_model.pt"

    )



    checkpoint = torch.load(

        checkpoint_path,

        map_location=config.device,

    )



    model.load_state_dict(

        checkpoint["model_state"]

    )



    model.to(

        config.device

    )



    model.eval()



    print(
        "Model loaded."
    )


    return model





# ============================================================
# Prepare molecule
# ============================================================


def create_molecule(
    atomic_numbers,
    coordinates,
):


    """
    Convert molecule input
    into PyTorch Geometric Data.
    """


    z = torch.tensor(

        atomic_numbers,

        dtype=torch.long,

    )



    pos = torch.tensor(

        coordinates,

        dtype=torch.float32,

    )



    batch = torch.zeros(

        len(z),

        dtype=torch.long,

    )



    data = Data(

        z=z,

        pos=pos,

        batch=batch,

    )



    return data





# ============================================================
# Prediction
# ============================================================


def predict(

    model,

    data,

):


    """
    Predict energy and forces.
    """


    data = data.to(

        config.device

    )



    energy, forces = model(

        data,

        return_forces=True,

    )



    return (

        energy.detach()
        .cpu(),

        forces.detach()
        .cpu(),

    )





# ============================================================
# Example molecule
# ============================================================


def main():


    print("="*60)

    print(
        "Molecular Prediction"
    )

    print("="*60)



    model = load_model()



    # Example:

    # Water molecule

    atomic_numbers = [

        8,

        1,

        1,

    ]



    coordinates = [

        [0.000, 0.000, 0.000],

        [0.758, 0.000, 0.504],

        [-0.758,0.000,0.504],

    ]



    molecule = create_molecule(

        atomic_numbers,

        coordinates,

    )



    energy, forces = predict(

        model,

        molecule,

    )



    print()

    print(
        "Predicted Energy:"
    )


    print(

        energy.item()

    )



    print()

    print(
        "Predicted Forces:"
    )



    print(forces)





if __name__ == "__main__":

    main()
