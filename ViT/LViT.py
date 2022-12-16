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
from vit_pytorch.efficient import ViT
from linformer import Linformer
from torchvision.io import read_image
from PIL import Image
from data_preprocess import get_data_loader
from tqdm import tqdm
import time as timer
# print("Finish importing packges")

trans = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42


train_loader, val_loader, test_loader = get_data_loader(batch_size,trans)

# print("Got data loaders.")

device = 'cuda'

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

model.load_state_dict(torch.load("initial.pt"))




# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)
# scheduler
# scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


train_acc = []
val_acc = []
test_acc = []
for epoch in range(epochs):
    print(f"===== Epoch : {epoch+1} ======\n", flush = True)
    start = timer.time()
    train_loss = 0
    train_accuracy = 0
    print("Training:", flush = True)
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean().item()
        train_accuracy += acc / len(train_loader)
        train_loss += loss.item() / len(train_loader)
        # scheduler.step()
        
    with torch.no_grad():
        val_accuracy = 0
        val_loss = 0
        print("Validation:", flush = True)
        for data, label in tqdm(val_loader):
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean().item()
            val_accuracy += acc / len(val_loader)
            val_loss += loss.item() / len(val_loader)
            
            
        test_accuracy = 0
        test_loss = 0
        print("Test:", flush = True)
        for data, label in tqdm(test_loader):
            data = data.cuda()
            label = label.cuda()

            test_output = model(data)
            loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean().item()
            test_accuracy += acc / len(test_loader)
            test_loss += loss.item() / len(test_loader)
    
    end = timer.time()
    print(
        f"Epoch : {epoch+1} - training loss : {train_loss:.4f} -  training acc: {train_accuracy:.4f} - val_loss : {val_loss:.4f} - val_acc: {val_accuracy:.4f}- test_loss : {test_loss:.4f} - test_acc: {test_accuracy:.4f} - time: {end - start: 4f}\n", flush = True)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    test_acc.append(test_accuracy)
    model_name = "ViT_efficient_" + str(epoch+1) + "TE.pt"
    
    if test_accuracy > 0.85:
        torch.save(model.state_dict(), model_name)
model_name = "ViT_efficient_" + str(epoch+1) + "TE.pt"
torch.save(model.state_dict(), model_name)

dic = {"train_acc":train_acc,"val_acc":val_acc,"test_acc":test_acc}
df = pd.DataFrame(dic) 
df.to_csv('acc_effViT.csv') 



