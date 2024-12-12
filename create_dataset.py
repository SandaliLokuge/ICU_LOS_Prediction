"""
Combine data into a single dataset
"""

from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pandas as pd 

class ICUDataset(Dataset):
    def __init__(self, clinical_data, img_data, icu_stay):
        self.clinical_data = torch.tensor(clinical_data.values, dtype=torch.float32)
        self.img_data = torch.tensor(img_data.values, dtype=torch.float32)
        self.icu_stay = torch.tensor(icu_stay.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.icu_stay)

    def __getitem__(self, idx):
        return self.clinical_data[idx], self.img_data[idx], self.icu_stay[idx]