import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize
import torchvision
from vit_pytorch.efficient import ViT
from linformer import Linformer
from tqdm.notebook import tqdm
from torchvision.io import read_image
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')

        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        # print(type(image))
        if self.transform:
            image = self.transform(image)
            # print(image.size())

        return image, y_label

def get_data_loader(batch_size,trans):
    train_df = pd.read_csv('data/train.txt', sep=" ", header=None)
    train_df.columns=['patient id', 'filename', 'class', 'data source']
    train_df=train_df.drop(['patient id', 'data source'], axis=1 )
    train_df['class'] = train_df['class'].map({'positive': 1, 'negative': 0})

    test_df = pd.read_csv('data/test.txt', sep=" ", header=None)
    test_df.columns=['id', 'filename', 'class', 'data source' ]
    test_df=test_df.drop(['id', 'data source'], axis=1 )
    test_df['class'] = test_df['class'].map({'positive': 1, 'negative': 0})

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42) 

    


    

    trainset = CustomDataset(train_df, 'data/train', transform=trans)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers = 8)

    valset = CustomDataset(val_df, 'data/train', transform=trans)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True,num_workers = 8)

    testset = CustomDataset(test_df, 'data/test', transform=trans)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True,num_workers = 8)


    return train_loader, val_loader, test_loader
