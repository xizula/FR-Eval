import torch
from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

class FaceRecoDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, folder_mapping=None):
        super().__init__(root, transform=transform)
        self.class_to_idx = folder_mapping

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        image, label = original_tuple
        path, _ = self.samples[index]

        return image, label, path
    
    def get_folder_to_label_mapping(self):
        return {v: k for k, v in self.class_to_idx.items()}


def get_dataset(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_path = Path(config['data_path'])
    # dataset_name = dataset_name + "_images"
    dataset_dir = data_path / dataset_name
    subfolders = sorted(os.listdir(dataset_dir))
    folder_to_label = {folder: idx for idx, folder in enumerate(subfolders)}

    dataset = FaceRecoDataset(root=dataset_dir, transform=transform, folder_mapping=folder_to_label)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader
