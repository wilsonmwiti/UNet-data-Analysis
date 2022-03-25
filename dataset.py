import torch
import json
import SimpleITK as sitk
import numpy as np

from torch.utils.data import Dataset

def collate_fn(data):
    images, labels = [], []
    
    for image, label in data:
        images.append(image)
        labels.append(label)
    
    images = torch.vstack(images).unsqueeze(1)
    labels = torch.vstack(labels).unsqueeze(1)
    
    return images, labels

class ACDCDataset(Dataset):
    def __init__(self, image_list, label_list, data_json_path):
        self.image_list = image_list
        self.label_list = label_list
        with open(data_json_path, 'r') as f:
            self.data_json = json.load(f)
        
        
    def __getitem__(self, index):
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        
        resize = 224
        image_array = self._get_array_from_nii(image_path)
        image_array = np.resize(image_array, (image_array.shape[0], resize, resize)) # [depth, heigth, width]
        label_array = (self._get_array_from_nii(label_path) > 0).astype(np.float)
        label_array = np.resize(label_array,(label_array.shape[0], resize, resize)) # [depth, heigth, width]
        
        return torch.from_numpy(image_array), torch.from_numpy(label_array)
        
    def _get_array_from_nii(self, nii_file):
        array = sitk.ReadImage(nii_file)
        array = sitk.GetArrayFromImage(array)
        return array
    
    def __len__(self):
        return len(self.image_list)