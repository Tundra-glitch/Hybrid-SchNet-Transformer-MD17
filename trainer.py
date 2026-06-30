"""
trainer.py

Training engine for Hybrid SchNet + Multi-Head Transformer.

Handles:

- Training loop
- Validation loop
- Optimization
- Scheduler
- Checkpointing
- Early stopping
- Metrics

Project:
Hybrid-SchNet-Transformer-MD17
"""


from pathlib import Path

from typing import Dict


import torch

from torch import Tensor

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR


from tqdm import tqdm


from config import config

from loss import MolecularLoss, calculate_metrics




# ============================================================
# Trainer Class
# ============================================================


class Trainer:
    """
    Research-grade training engine.
    """



    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
    ):


        self.model = model.to(
            config.device
        )


        self.train_loader = train_loader

        self.validation_loader = validation_loader



        self.criterion = MolecularLoss()



        self.optimizer = AdamW(

            self.model.parameters(),

            lr=config.learning_rate,

            weight_decay=config.weight_decay,

        )



        self.scheduler = CosineAnnealingLR(

            self.optimizer,

            T_max=config.epochs,

            eta_min=config.minimum_learning_rate,

        )



        self.best_loss = float("inf")


        self.counter = 0




        Path(
            config.checkpoint_dir
        ).mkdir(
            exist_ok=True
        )




    # ========================================================
    # Training One Epoch
    # ========================================================


    def train_epoch(self) -> Dict[str,float]:


        self.model.train()


        total_loss = 0.0


        energy_loss_total = 0.0


        force_loss_total = 0.0



        progress = tqdm(

            self.train_loader,

            desc="Training"

        )



        for batch in progress:


            batch = batch.to(
                config.device
            )



            self.optimizer.zero_grad()



            energy, forces = self.model(

                batch,

                return_forces=True,

            )



            losses = self.criterion(

                energy,

                batch.y,

                forces,

                batch.force,

            )



            loss = losses["loss"]



            loss.backward()



            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                config.gradient_clip,

            )



            self.optimizer.step()



            total_loss += loss.item()


            energy_loss_total += (
                losses["energy_loss"]
                .item()
            )


            force_loss_total += (
                losses["force_loss"]
                .item()
            )



            progress.set_postfix(

                loss=f"{loss.item():.4f}"

            )




        return {


            "loss":

            total_loss
            /
            len(self.train_loader),



            "energy_loss":

            energy_loss_total
            /
            len(self.train_loader),



            "force_loss":

            force_loss_total
            /
            len(self.train_loader),

        }




    # ========================================================
    # Validation
    # ========================================================


    @torch.no_grad()

    def validate(self) -> Dict[str,float]:


        self.model.eval()



        total_loss = 0.0



        all_energy_pred = []

        all_energy_true = []

        all_force_pred = []

        all_force_true = []



        for batch in tqdm(

            self.validation_loader,

            desc="Validation"

        ):


            batch = batch.to(
                config.device
            )



            energy, forces = self.model(

                batch,

                return_forces=False,

            )



            losses = self.criterion(

                energy,

                batch.y,

                forces,

                batch.force,

            )



            total_loss += (
                losses["loss"]
                .item()
            )



            all_energy_pred.append(

                energy.cpu()

            )


            all_energy_true.append(

                batch.y.cpu()

            )



        energy_pred = torch.cat(

            all_energy_pred

        )


        energy_true = torch.cat(

            all_energy_true

        )



        metrics = {


            "loss":

            total_loss
            /
            len(self.validation_loader),


            "energy_MAE":

            torch.mean(

                torch.abs(

                    energy_pred
                    -
                    energy_true

                )

            ).item(),

        }



        return metrics




    # ========================================================
    # Save Model
    # ========================================================


    def save_checkpoint(
        self,
        epoch:int,
        loss:float,

    ):


        path = (

            Path(config.checkpoint_dir)

            /

            "best_model.pt"

        )



        torch.save(

            {

            "epoch":epoch,

            "model_state":

            self.model.state_dict(),


            "optimizer_state":

            self.optimizer.state_dict(),


            "loss":loss,

            },

            path,

        )



    # ========================================================
    # Full Training
    # ========================================================


    def fit(self):


        history = []



        for epoch in range(

            config.epochs

        ):


            print(

                f"\nEpoch {epoch+1}/{config.epochs}"

            )



            train_metrics = (
                self.train_epoch()
            )



            val_metrics = (
                self.validate()
            )



            self.scheduler.step()



            print(

                "Train:",

                train_metrics

            )


            print(

                "Validation:",

                val_metrics

            )



            history.append(

                {

                "epoch":epoch,

                **train_metrics,

                **val_metrics,

                }

            )



            val_loss = (

                val_metrics["loss"]

            )



            if val_loss < self.best_loss:


                self.best_loss = val_loss


                self.counter = 0



                self.save_checkpoint(

                    epoch,

                    val_loss,

                )


            else:


                self.counter += 1



            if (

                self.counter
                >
                config.early_stopping_patience

            ):


                print(
                    "Early stopping."
                )


                break



        return history




# ============================================================
# End
# ============================================================
