import torch
import typer
from model import MyAwesomeModel
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.manifold import TSNE

def get_data(data_folder):
    images = torch.load(data_folder / 'train_images.pt')
    targets = torch.load(data_folder / 'train_targets.pt')
    return images, targets

def visualize(data_path:Path = Path('data/processed'), model_checkpoint:str="trained_model.pth"):
    model_checkpoint = "models/"+model_checkpoint
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    images, targets = get_data(data_folder=data_path)
    images = images[:4000,:,:,:]
    targets = targets[:4000]
    N = images.shape[0]
    idx = random.randint(0,N-1)
    image = images[idx, :, :, :]
    # print(f"Index {idx}")
    image = image.unsqueeze(0)
    # print(image.shape)
    model.eval()
    with torch.no_grad():
        rep = model.get_intermediate_representation(image)
        print(rep)

    model.eval()
    with torch.no_grad():
        intermediate_representations = model.get_intermediate_representation(images)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(intermediate_representations)

    # Plot the 2D t-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=targets, cmap='viridis', s=5)
    plt.colorbar()
    plt.title('t-SNE of Intermediate Representations')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('reports/figures/t-SNE.png')
    plt.show()
    

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    typer.run(visualize)

    