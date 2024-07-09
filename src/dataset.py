import pandas as pd
import torch
from torch.utils.data import Dataset
import io

class PotatoDataset(Dataset):

    def __init__(
            self,
            image_meta_path: str,
    ):
        self.dataset_meta = pd.read_csv(image_meta_path)

    def __len__(self):
        return len(self.dataset_meta)
    
    def __getitem__(self, idx):
        sample = self.dataset_meta.iloc[idx]
        image_path = sample['image_path']
        label = sample['label']
        return image_path, label
    
def collate_function(batch):
    image_paths, labels = zip(*batch)
    image = torch.Tensor([
        torch.Tensor(
            io.imread(image_path) for image_path in image_paths
        ).float()
    ])
    labels = torch.Tensor(labels).long()
    return image, labels
    