"""
dataset.py

Dataset utilities for Revised MD17 molecular dynamics dataset.

This module handles:
- Dataset loading
- Graph representation
- Train/validation/test splitting
- DataLoader creation

Project:
Hybrid SchNet + Multi-Head Transformer for Molecular Energy
and Force Prediction
"""


from typing import Tuple

import torch

from torch.utils.data import random_split

from torch_geometric.datasets import MD17

from torch_geometric.loader import DataLoader

from torch_geometric.data import Dataset


from config import config



# ============================================================
# Dataset Loading
# ============================================================


def load_md17_dataset() -> Dataset:
    """
    Load Revised MD17 dataset.

    Returns
    -------
    Dataset
        PyTorch Geometric molecular dataset.
    """

    dataset = MD17(
        root=str(config.data_dir),
        name=config.molecule,
    )

    return dataset



# ============================================================
# Dataset Split
# ============================================================


def split_dataset(
    dataset: Dataset,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation and test sets.

    Parameters
    ----------
    dataset:
        Full molecular dataset.

    Returns
    -------
    train_dataset
    validation_dataset
    test_dataset
    """

    total_size = len(dataset)


    train_size = int(
        total_size * config.train_ratio
    )


    validation_size = int(
        total_size * config.validation_ratio
    )


    test_size = (
        total_size
        - train_size
        - validation_size
    )


    generator = torch.Generator()

    generator.manual_seed(
        config.random_seed
    )


    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [
            train_size,
            validation_size,
            test_size,
        ],
        generator=generator,
    )


    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )



# ============================================================
# DataLoader Creation
# ============================================================


def create_dataloaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
):
    """
    Create PyTorch Geometric DataLoaders.

    Returns
    -------
    train_loader
    validation_loader
    test_loader
    """


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


    return (
        train_loader,
        validation_loader,
        test_loader,
    )



# ============================================================
# Quick Test
# ============================================================


def test_dataset():
    """
    Simple dataset sanity check.
    """


    dataset = load_md17_dataset()


    print(dataset)

    print(
        "Number of molecules:",
        len(dataset)
    )


    sample = dataset[0]


    print(sample)


    print(
        "Atomic numbers:",
        sample.z.shape
    )


    print(
        "Coordinates:",
        sample.pos.shape
    )


    print(
        "Energy:",
        sample.y.shape
    )


    print(
        "Forces:",
        sample.force.shape
    )



if __name__ == "__main__":

    test_dataset()
