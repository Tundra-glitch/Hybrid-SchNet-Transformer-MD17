"""
test.py

Sanity tests for Hybrid SchNet + Multi-Head Transformer.

Before training, this script checks:

1. Dataset
2. DataLoader
3. Model forward pass
4. Energy prediction
5. Force calculation
6. Loss calculation
7. Backpropagation

Run:

python test.py


Project:
Hybrid-SchNet-Transformer-MD17
"""


import torch



from config import config


from dataset import (
    load_md17_dataset,
    split_dataset,
    create_dataloaders,
)


from model import HybridSchNetTransformer


from loss import MolecularLoss





# ============================================================
# Test Configuration
# ============================================================


def test_config():

    print("\nTesting configuration...")


    print(
        "Project:",
        config.project_name
    )


    print(
        "Device:",
        config.device
    )


    print("✓ Config OK")





# ============================================================
# Test Dataset
# ============================================================


def test_dataset():

    print("\nTesting dataset...")


    dataset = load_md17_dataset()


    assert len(dataset) > 0


    sample = dataset[0]


    print(sample)


    assert hasattr(
        sample,
        "z"
    )


    assert hasattr(
        sample,
        "pos"
    )


    assert hasattr(
        sample,
        "force"
    )


    print("✓ Dataset OK")


    return dataset





# ============================================================
# Test DataLoader
# ============================================================


def test_dataloader(dataset):

    print("\nTesting DataLoader...")


    train_dataset, val_dataset, test_dataset = (

        split_dataset(dataset)

    )


    train_loader, _, _ = (

        create_dataloaders(

            train_dataset,

            val_dataset,

            test_dataset,

        )

    )


    batch = next(
        iter(train_loader)
    )


    print(batch)



    print(
        "Atoms:",
        batch.z.shape
    )


    print(
        "Positions:",
        batch.pos.shape
    )


    print(
        "Energy:",
        batch.y.shape
    )


    print(
        "Forces:",
        batch.force.shape
    )


    print("✓ DataLoader OK")


    return batch





# ============================================================
# Test Model
# ============================================================


def test_model(batch):

    print("\nTesting model...")


    model = HybridSchNetTransformer()


    model = model.to(
        config.device
    )


    batch = batch.to(
        config.device
    )



    energy, forces = model(

        batch,

        return_forces=True

    )



    print(

        "Energy output:",

        energy.shape

    )


    print(

        "Force output:",

        forces.shape

    )



    assert energy is not None


    assert forces is not None



    print("✓ Forward pass OK")



    return (

        model,

        energy,

        forces,

    )





# ============================================================
# Test Loss
# ============================================================


def test_loss(
    model,
    batch,
    energy,
    forces,
):

    print("\nTesting loss...")



    criterion = MolecularLoss()



    output = criterion(

        energy,

        batch.y,

        forces,

        batch.force,

    )


    print(output)



    loss = output["loss"]


    assert torch.isfinite(loss)



    print(
        "Loss:",
        loss.item()
    )


    print("✓ Loss OK")


    return loss





# ============================================================
# Test Backpropagation
# ============================================================


def test_backward(
    model,
    loss,
):

    print("\nTesting backward...")


    loss.backward()



    gradients_exist = False



    for parameter in model.parameters():


        if parameter.grad is not None:


            gradients_exist = True


            break




    assert gradients_exist


    print("✓ Backpropagation OK")





# ============================================================
# Main
# ============================================================


def main():


    print("="*60)

    print(
        "Running project tests"
    )

    print("="*60)



    test_config()



    dataset = test_dataset()



    batch = test_dataloader(
        dataset
    )



    model, energy, forces = test_model(
        batch
    )



    loss = test_loss(

        model,

        batch,

        energy,

        forces,

    )



    test_backward(

        model,

        loss,

    )



    print("\n" + "="*60)

    print(
        "ALL TESTS PASSED SUCCESSFULLY"
    )

    print("="*60)





if __name__ == "__main__":

    main()
