import torch
import typer
from model import MyAwesomeModel
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

def get_data(data_folder: Path = Path('data/processed')):
    images = torch.load(data_folder / 'test_images.pt')
    targets = torch.load(data_folder / 'test_targets.pt')
    dataset = TensorDataset(images, targets)
    return dataset

def evaluate_model(data_path:Path = Path('data/processed'), model_checkpoint:str="trained_model.pth"):
    model_checkpoint = "models/"+model_checkpoint
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    test_set = get_data(data_folder=data_path)
    
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
    warnings.filterwarnings("ignore", category=FutureWarning)
    typer.run(evaluate_model)
    warnings.filterwarnings("default", category=FutureWarning)
