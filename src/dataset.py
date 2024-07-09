from typing import Optional
import os

import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader

from transform import ImageTransform

class PotatoDataset(Dataset):

    def __init__(
            self,
            image_meta_path: str,
            image_data_path: str,
            split: str = "all",
            transform: Optional[ImageTransform] = None
    ):
        self.image_data_path = image_data_path
        self.dataset_meta = pd.read_csv(image_meta_path)
        if split != "all":
            self.dataset_meta = self.dataset_meta[self.dataset_meta['split'] == split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_meta)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset_meta.iloc[idx]
        image_path = sample['image']
        label = sample['label']

        image_path = os.path.join(
            self.image_data_path,
            image_path
        )
        image = io.imread(image_path)

        sample = {
            'image': image,
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    

def dataset_factory(
        image_meta_path: str, 
        image_data_path: str,
        split: str = "train", 
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        transform = ImageTransform(
            output_size=[255, 255], rescale=True, random_crop=0.2, to_tensor=True
        )
    ):
    dataset = PotatoDataset(image_meta_path, image_data_path, split, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, data_loader



if __name__=="__main__":
    image_meta_path = "potato-plant-diseases-data/PlantVillage/dataset_metadata.csv"
    image_data_path = "potato-plant-diseases-data/PlantVillage/PlantVillage"
    split = "train"

    transform = ImageTransform(
        output_size=[255, 255], rescale=True, random_crop=0.2, to_tensor=True
    )
    dataset = PotatoDataset(image_meta_path, image_data_path, split, transform)
    print(len(dataset))
    for i in range(10):
        sample = dataset[i]
        print(sample['image'].shape)
        print(sample['label'])
        print("===")

    _, data_loader = dataset_factory(image_meta_path, image_data_path, split, transform=transform)
    for i, batch in enumerate(data_loader):
        print(batch["image"][0].shape)
        print(batch["label"])
        break