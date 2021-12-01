import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, inputs, labels) :
        super().__init__()
        self.inputs = inputs 
        self.labels = labels 
    
    def __len__(self) :
        return len(self.labels)
    
    def __getitem__(self, index):
        y = self.labels[index]
        x = self.inputs[index]
        return x, y 

