import torch
import typer
from model import MyAwesomeModel
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import wandb
from PIL import Image

def get_data(data_folder: Path = Path('data/processed')):
    images = torch.load('/gcs/rita-sucks/processed/train_images.pt')
    targets = torch.load('/gcs/rita-sucks/processed/train_images.pt')
    dataset = TensorDataset(images, targets)
    return dataset

def train_model(lr:float = 1e-3, epochs:int = 7, data_folder: Path = Path('data/processed')):
    dataset = get_data(data_folder)
    print("Training data loaded succesfully")
    model  = MyAwesomeModel()
    print("Model loaded succesfully")
    batch_size = 64
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_input = torch.rand((2,1,28,28))
    model.forward(test_input)
    print("Test pass succesful")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    losses = []
    accs = []

    run = wandb.init(
    project="MLOps",
    name="Logging a model",
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size":batch_size
    },
    )
    for _ in range(epochs):
        loss_in_epoch = 0
        items_checked = 0
        accuracy = 0
        for images, labels in trainloader:
            out = model.forward(images)
            loss = criterion(out, labels)
            loss_in_epoch += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, predcited_classes = torch.topk(out, 1, dim = 1)

            equals = predcited_classes == labels.view(*predcited_classes.shape)
            accuracy += torch.sum(equals)
            items_checked += labels.shape[0]

        print(f"Loss is {loss_in_epoch:.4f}")
        losses.append(loss_in_epoch)
        
        accs.append(accuracy / items_checked)
        wandb.log({"accuracy": accuracy / items_checked, "loss": loss_in_epoch})
    

    trained_path = "models"
    trained_model_name = "trained_model.pth"
    tm = trained_path + "/" + trained_model_name
    torch.save(model.state_dict(), tm)
    print(f"Saved model to {trained_path} as {trained_model_name}")

    art = wandb.Artifact(
        name = "CNN-model",
        type = "model",
        description = "My model",
        metadata = dict(run.config)
        )
    art.add_file(local_path = tm, name = "A_trained_model")
    art.save()

    figs_folder = "reports/figures"
    fig_name = "train_curve.png"
    f = figs_folder + "/" + fig_name
    epochs = [i for i in range(1, epochs+1)]
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot losses on the primary y-axis
    ax1.plot(epochs, losses, 'b-', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CrossEntropyLoss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a secondary y-axis to plot accuracies
    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, 'r-', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add legends
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    # Save the figure
    plt.savefig(f)
    print(f"Saved training figure to {figs_folder} as {fig_name}")

    wandb.log({"Image": fig})
    
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("Creating a trained model")
    typer.run(train_model)
    warnings.filterwarnings("default", category=FutureWarning)
    