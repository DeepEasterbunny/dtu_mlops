from pathlib import Path

import typer
from torch.utils.data import Dataset
import torch
import warnings

app = typer.Typer()

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.train_images, self.train_targets = self.load_training_data()
        self.test_images, self.test_targets = self.load_test_data()

    def load_training_data(self):
        images = []
        targets = []
        image_files = list(self.data_path.glob("train_images_*.pt"))
        target_files = list(self.data_path.glob("train_target_*.pt"))
        
        if not image_files or not target_files:
            raise FileNotFoundError("No training data files found in the specified path.")
        
        warnings.filterwarnings("ignore", category=FutureWarning)
        for file in self.data_path.glob("train_images_*.pt"):
            images.append(torch.load(file))
        for file in self.data_path.glob("train_target_*.pt"):
            targets.append(torch.load(file))
        warnings.filterwarnings("default", category=FutureWarning)

        return images, targets
    
    def load_test_data(self):
        images = []
        targets = []
        image_files = list(self.data_path.glob("test_images.pt"))
        target_files = list(self.data_path.glob("test_target.pt"))
        
        if not image_files or not target_files:
            raise FileNotFoundError("No test data files found in the specified path.")
        
        warnings.filterwarnings("ignore", category=FutureWarning)
        for file in self.data_path.glob("test_images.pt"):
            images.append(torch.load(file))
        for file in self.data_path.glob("test_target.pt"):
            targets.append(torch.load(file))
        warnings.filterwarnings("default", category=FutureWarning)

        return images, targets
        

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.images[index], self.targets[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        test_images, test_targets = torch.cat(self.test_images), torch.cat(self.test_targets)
        train_images, train_targets = torch.cat(self.train_images), torch.cat(self.train_targets)


        train_images = train_images.view(-1, 28, 28)  
        train_max_values = train_images.view(train_images.size(0), -1).max(dim=1, keepdim=True)[0]
        train_images = train_images / train_max_values.view(-1, 1, 1)

        test_images = test_images.view(-1, 28, 28)
        test_max_values = test_images.view(test_images.size(0), -1).max(dim=1, keepdim=True)[0]
        test_images = test_images / test_max_values.view(-1, 1, 1)

        train_images = train_images.unsqueeze(1)
        test_images = test_images.unsqueeze(1)
        
        print(f"Loaded {self.train_targets[0].shape[0] * len(self.train_targets)} training images")
        print(f"Loaded {self.test_targets[0].shape[0]} test images")

        output_folder.mkdir(parents=True, exist_ok=True)
        torch.save(train_images, output_folder / 'train_images.pt')
        torch.save(train_targets, output_folder / 'train_targets.pt')
        torch.save(test_images, output_folder / 'test_images.pt')
        torch.save(test_targets, output_folder / 'test_targets.pt')

@app.command()
#def preprocess(raw_data_path: Path = Path('data/raw'), output_folder: Path= Path('data/processed')) -> None:
def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)
    print(f"Data saved to {output_folder}")


if __name__ == "__main__":
    app()
