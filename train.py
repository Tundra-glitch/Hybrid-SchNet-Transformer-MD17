
"""
train.py

Main training entry point.

Run:

python train.py

This script automatically:

1. Loads Revised MD17 dataset
2. Creates DataLoaders
3. Builds Hybrid SchNet Transformer model
4. Starts training
5. Saves best checkpoint

Project:
Hybrid-SchNet-Transformer-MD17
"""


import random

import numpy as np

import torch



from config import config


from dataset import (
    load_md17_dataset,
    split_dataset,
    create_dataloaders,
)


from model import HybridSchNetTransformer


from trainer import Trainer





# ============================================================
# Reproducibility
# ============================================================


def set_seed(
    seed:int
):
    """
    Set random seeds.
    """


    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )





# ============================================================
# Main
# ============================================================


def main():

    """
    Complete training pipeline.
    """


    print("="*60)

    print(
        config.project_name
    )

    print("="*60)



    print(

        "Device:",

        config.device

    )



    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(

        config.random_seed

    )



    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------


    print(
        "\nLoading dataset..."
    )


    dataset = load_md17_dataset()



    print(

        "Dataset size:",

        len(dataset)

    )



    train_dataset, validation_dataset, test_dataset = (

        split_dataset(dataset)

    )




    train_loader, validation_loader, test_loader = (

        create_dataloaders(

            train_dataset,

            validation_dataset,

            test_dataset,

        )

    )



    print(

        "Train:",
        len(train_dataset),

        "Validation:",
        len(validation_dataset),

        "Test:",
        len(test_dataset)

    )



    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------


    print(

        "\nBuilding model..."

    )


    model = HybridSchNetTransformer()



    print(model)




    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------


    trainer = Trainer(

        model,

        train_loader,

        validation_loader,

    )



    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------


    print(

        "\nStarting training..."

    )



    history = trainer.fit()



    print(

        "\nTraining finished."

    )


    print(

        "Best validation loss:",

        trainer.best_loss

    )





# ============================================================
# Execute
# ============================================================


if __name__ == "__main__":

    main()
