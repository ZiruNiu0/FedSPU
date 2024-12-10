from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd

class voice_dataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transforms.Normalize(mean=(0.5), std=(0.5))
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        x = torch.tensor(self.dataframe.iloc[index, :-1], dtype=torch.float32)
        y = torch.tensor(self.dataframe.iloc[index, -1])
        x = x.reshape([1,32,32])
        x = self.transform(x)
        return x, y
