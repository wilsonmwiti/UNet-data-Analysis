from pickletools import optimize
import tqdm
from model import UNet2D
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ACDCDataset, collate_fn
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_file_list(dir_path):
    filenames = os.listdir(os.path.join(dir_path))
    filelist = []
    for filename in filenames:
        filelist.append(os.path.join(dir_path,filename))
    return filelist

def train(epochs,dataloader, model, criterion, optimizer):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        dataset_size = len(dataloader.dataset)
        epoch_loss = 0
        step = 0
        for img, target in dataloader:
            step += 1
            img = img.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            print("backward...")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("{}/{}, train_loss:{}".format(step, (dataset_size - 1)//dataloader.batch_size, loss.item()))
        print("epoch %d loss:%0.3f"%(epoch, epoch_loss/step))
    torch.save({'model':model.state_dict, 'optimizer':optimizer.state_dict(), 'epoch':epoch}, "trained_models/checkpoint_"+str(epoch))
    return model

def main():
    epochs = 10

    path = "C:\\Users\\z1023\\WorkSpace\\biomedical\\nnUNetFrame\\UNet\\Task027_ACDC"
    image_list = get_file_list(os.path.join(path,'imagesTr'))
    label_list = get_file_list(os.path.join(path,'labelsTr'))
    data_json_path = os.path.join(path, "dataset.json")
    dataset =  ACDCDataset(image_list, label_list, data_json_path)
    dataloader = DataLoader(dataset, batch_size = 2, shuffle = True, collate_fn=collate_fn)
    
    model = UNet2D()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    train(epochs, dataloader, model, criterion, optimizer)
    
if __name__ == '__main__':
    main()
    