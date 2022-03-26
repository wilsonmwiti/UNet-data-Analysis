from pickletools import optimize
import tqdm
from model import UNet2D
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ACDCDataset, collate_fn
import os
import numpy as np
from train import get_file_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    
    intersection = (output * target).sum()
    
    return (2. * intersection + smooth)/ \
        (output.sum() + target.sum() + smooth)
    
def evaluate(dataloader, model, checkpoint_path):
    model.to(device)
    model.eval()
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model'])
    
    dice_coefs = []
    
    with torch.no_grad():
        for img, target in dataloader:
            num_slices = img.shape[0]
            img = img.to(device)
            target = target.to(device).squeeze(1)
            output = model(img).squeeze(1)
            dice_coefs += [dice_coef(target[i],target[i]) for i in range(num_slices)]
            print(dice_coefs)
    
    return np.mean(dice_coefs)

def main():
    epochs = 10

    path = "./Task027_ACDC"
    image_list = get_file_list(os.path.join(path,'imagesTr'))
    image_list.sort()
    label_list = get_file_list(os.path.join(path,'labelsTr'))
    label_list.sort()
    data_json_path = os.path.join(path, "dataset.json")
    dataset =  ACDCDataset(image_list, label_list, data_json_path)
    dataloader = DataLoader(dataset, batch_size = 2, shuffle = True, collate_fn=collate_fn)
    
    model = UNet2D()
    
    evaluate(dataloader, model, 'checkpoint_150')
    
if __name__ == '__main__':
    main()
    