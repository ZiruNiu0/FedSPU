import os
import pandas as pd
import torch
import random
import csv
import json
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict, defaultdict

class EmnistDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        x = torch.tensor(self.dataframe.iloc[index, :-1], dtype=torch.float32)
        y = torch.tensor(self.dataframe.iloc[index, -1])
        x = x.reshape([1,28,28])
        return x, y

