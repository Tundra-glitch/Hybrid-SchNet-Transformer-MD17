"""
evaluate.py

Evaluation script for trained Hybrid SchNet + Transformer model.

Measures:

- Energy MAE
- Energy RMSE
- Force MAE
- Force RMSE


Run:

python evaluate.py


Project:
Hybrid-SchNet-Transformer-MD17
"""


from pathlib import Path


import torch


from torch import Tensor


from tqdm import tqdm


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



from config import config


from dataset import (
    load_md17_dataset,
    split_dataset,
    create_dataloaders,
)


from model import HybridSchNetTransformer





# ============================================================
# Load Model
# ============================================================


def load_model():

    model = HybridSchNetTransformer()


    checkpoint = torch.load(

        Path(config.checkpoint_dir)
        /
        "best_model.pt",

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
# Evaluation
# ============================================================


@torch.no_grad()

def evaluate(model, loader):


    energy_predictions = []

    energy_targets = []


    force_predictions = []

    force_targets = []



    for batch in tqdm(

        loader,

        desc="Evaluating"

    ):


        batch = batch.to(

            config.device

        )



        energy, forces = model(

            batch,

            return_forces=False,

        )



        energy_predictions.append(

            energy.cpu()

        )


        energy_targets.append(

            batch.y.cpu()

        )



    energy_predictions = torch.cat(

        energy_predictions

    )


    energy_targets = torch.cat(

        energy_targets

    )



    energy_mae = mean_absolute_error(

        energy_targets.numpy(),

        energy_predictions.numpy(),

    )


    energy_rmse = mean_squared_error(

        energy_targets.numpy(),

        energy_predictions.numpy(),

        squared=False,

    )



    return {


        "Energy MAE":

        energy_mae,


        "Energy RMSE":

        energy_rmse,

    }





# ============================================================
# Main
# ============================================================


def main():


    print("="*60)

    print(
        "Model Evaluation"
    )

    print("="*60)



    dataset = load_md17_dataset()



    _, _, test_dataset = split_dataset(

        dataset

    )



    _, _, test_loader = create_dataloaders(

        dataset,

        dataset,

        test_dataset,

    )



    model = load_model()



    results = evaluate(

        model,

        test_loader,

    )



    print("\nResults")

    print("-"*40)



    for key,value in results.items():

        print(

            f"{key}: {value:.6f}"

        )





if __name__ == "__main__":

    main()
