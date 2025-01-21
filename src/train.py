import matplotlib.pyplot as plt
import torch
import typer

import wandb
from data_solution import corrupt_mnist
from model_solution import MyAwesomeModel
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning rate: {lr}, Batch size: {batch_size}, Epochs: {epochs}")

    # Initialize Weights & Biases
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(img)
            loss = loss_fn(y_pred, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy for logging
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            # Collect predictions and targets for later metrics
            preds.append(y_pred.detach().cpu())  # Move to CPU
            targets.append(target.detach().cpu())  # Move to CPU

            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item()}")

                # Log input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # Log histogram of gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.cpu())})  # Move gradients to CPU

        # Aggregate predictions and targets
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        # Log ROC curves
        plt.figure()
        for class_id in range(10):
            one_hot = (targets == class_id).float()
            RocCurveDisplay.from_predictions(
                one_hot.cpu().numpy(), preds[:, class_id].cpu().numpy(), name=f"Class {class_id}"
            )
        wandb.log({"roc_curve": plt})
        plt.close()

    # Calculate final metrics
    final_accuracy = accuracy_score(targets.numpy(), preds.argmax(dim=1).numpy())
    final_precision = precision_score(targets.numpy(), preds.argmax(dim=1).numpy(), average="weighted")
    final_recall = recall_score(targets.numpy(), preds.argmax(dim=1).numpy(), average="weighted")
    final_f1 = f1_score(targets.numpy(), preds.argmax(dim=1).numpy(), average="weighted")

    # Save and log the model as a W&B artifact
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={
            "accuracy": final_accuracy,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
        },
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    print("Training complete and model artifact logged!")

    run.finish()


if __name__ == "__main__":
    typer.run(train)
