"""
visualize.py

Visualization tools for Hybrid SchNet + Transformer.

Creates:

- Training curves
- Energy prediction plots
- Error distribution

Project:
Hybrid-SchNet-Transformer-MD17
"""


from pathlib import Path


import torch


import matplotlib.pyplot as plt


import numpy as np



from config import config



# ============================================================
# Create folder
# ============================================================


Path(
    config.figure_dir
).mkdir(
    exist_ok=True
)




# ============================================================
# Training Curve
# ============================================================


def plot_training_history(
    history,
):

    """
    Plot training and validation loss.
    """


    epochs = [

        h["epoch"]

        for h in history

    ]



    train_loss = [

        h["loss"]

        for h in history

    ]



    val_loss = [

        h["loss"]

        for h in history

    ]



    plt.figure(
        figsize=(8,5)
    )


    plt.plot(

        epochs,

        train_loss,

        label="Training",

    )


    plt.plot(

        epochs,

        val_loss,

        label="Validation",

    )



    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Training Curve"
    )


    plt.legend()



    plt.savefig(

        Path(config.figure_dir)
        /
        "loss_curve.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()




# ============================================================
# Energy Prediction
# ============================================================


def plot_energy_prediction(

    true_energy,

    predicted_energy,

):


    """
    True vs predicted energy.
    """


    plt.figure(

        figsize=(6,6)

    )


    plt.scatter(

        true_energy,

        predicted_energy,

        alpha=0.5,

    )



    minimum = min(

        true_energy.min(),

        predicted_energy.min()

    )


    maximum = max(

        true_energy.max(),

        predicted_energy.max()

    )



    plt.plot(

        [minimum, maximum],

        [minimum, maximum],

    )



    plt.xlabel(

        "True Energy"

    )


    plt.ylabel(

        "Predicted Energy"

    )


    plt.title(

        "Energy Prediction"

    )



    plt.savefig(

        Path(config.figure_dir)
        /
        "energy_prediction.png",

        dpi=300,

        bbox_inches="tight",

    )



    plt.close()




# ============================================================
# Error Distribution
# ============================================================


def plot_error_distribution(

    true_energy,

    predicted_energy,

):


    error = (

        predicted_energy

        -

        true_energy

    )



    plt.figure(

        figsize=(8,5)

    )



    plt.hist(

        error,

        bins=50,

    )



    plt.xlabel(

        "Prediction Error"

    )


    plt.ylabel(

        "Frequency"

    )


    plt.title(

        "Energy Error Distribution"

    )



    plt.savefig(

        Path(config.figure_dir)
        /
        "energy_error_distribution.png",

        dpi=300,

        bbox_inches="tight",

    )


    plt.close()




# ============================================================
# Example Test
# ============================================================


if __name__ == "__main__":


    print(
        "Generating example plots..."
    )



    true = np.random.randn(

        100

    )



    prediction = (

        true

        +

        0.05*np.random.randn(100)

    )



    plot_energy_prediction(

        true,

        prediction,

    )


    plot_error_distribution(

        true,

        prediction,

    )



    print(
        "Figures saved."
    )
