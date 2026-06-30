"""
config.py

Central configuration for the Hybrid SchNet + Multi-Head Transformer project.

All configurable parameters should be defined here.

Author: Zhechen Wang
Project: Hybrid SchNet + Multi-Head Transformer for Molecular Energy
and Force Prediction on Revised MD17
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class Config:
    """Project configuration."""

    # ==========================================================
    # Project
    # ==========================================================
    project_name: str = "Hybrid-SchNet-Transformer-MD17"

    random_seed: int = 42

    # ==========================================================
    # Paths
    # ==========================================================
    root_dir: Path = Path.cwd()

    data_dir: Path = root_dir / "data"

    checkpoint_dir: Path = root_dir / "checkpoints"

    result_dir: Path = root_dir / "results"

    figure_dir: Path = root_dir / "figures"

    # ==========================================================
    # Dataset
    # ==========================================================
    dataset_name: str = "Revised MD17"

    molecule: str = "aspirin"

    train_ratio: float = 0.80

    validation_ratio: float = 0.10

    test_ratio: float = 0.10

    batch_size: int = 32

    num_workers: int = 4

    pin_memory: bool = True

    # ==========================================================
    # Model
    # ==========================================================
    hidden_dim: int = 128

    num_interactions: int = 6

    num_transformer_layers: int = 2

    num_attention_heads: int = 8

    dropout: float = 0.10

    activation: str = "SiLU"

    # ==========================================================
    # Training
    # ==========================================================
    epochs: int = 200

    learning_rate: float = 1e-4

    weight_decay: float = 1e-5

    gradient_clip: float = 1.0

    early_stopping_patience: int = 30

    # ==========================================================
    # Learning Rate Scheduler
    # ==========================================================
    scheduler: str = "CosineAnnealingLR"

    minimum_learning_rate: float = 1e-6

    # ==========================================================
    # Loss
    # ==========================================================
    energy_loss_weight: float = 1.0

    force_loss_weight: float = 100.0

    # ==========================================================
    # Device
    # ==========================================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================================
    # Logging
    # ==========================================================
    print_every: int = 10

    save_best_only: bool = True

    verbose: bool = True


config = Config()
