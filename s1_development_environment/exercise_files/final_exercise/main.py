import torch
import typer
from data_solution import corrupt_mnist
from model import MyAwesomeModel
import warnings
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    
    model = MyAwesomeModel()
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    train_set, _ = corrupt_mnist()
    warnings.filterwarnings("default", category=FutureWarning)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    test_input = torch.rand((2,1,28,28))
    model.forward(test_input)
    print("Test pass succesful")

    epochs = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    losses = []
    accs = []
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
    

    trained_path = "trained_models"
    trained_model_name = "trained_model.pth"
    tm = trained_path + "/" + trained_model_name
    torch.save(model.state_dict(), tm)
    print(f"Saved model to {trained_path} as {trained_model_name}")

    figs_folder = "figs"
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


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    model_checkpoint = "trained_models/"+model_checkpoint
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    warnings.filterwarnings("ignore", category=FutureWarning)
    _, test_set = corrupt_mnist()
    warnings.filterwarnings("default", category=FutureWarning)
    
    testloader  = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    with torch.no_grad():
        model.eval()
        items_checked = 0
        accuracy = 0
        for images, labels in testloader:
            log_ps = model(images)
            _, predcited_classes = torch.topk(log_ps, 1, dim = 1)

            equals = predcited_classes == labels.view(*predcited_classes.shape)
            accuracy += torch.sum(equals)
            items_checked += labels.shape[0]
        print(f"Validation accuracy: {accuracy/ items_checked * 100:.2f}%")  

if __name__ == "__main__":
    app()
