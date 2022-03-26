import json
import os
import pickle
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk

from train import get_file_list
from collect_intensity import IntensityAnalyzer


class DatasetAnalyzer(IntensityAnalyzer):
    def __init__(self, dataset_json, image_list, label_list):
        super(DatasetAnalyzer, self).__init__(dataset_json, image_list, label_list)
        
    def load_properties(self, img):
        properties = {}
        properties['size'] = img.GetSize()
        properties['spacing'] = img.GetSpacing()
        properties['origin'] = img.GetOrigin()
        properties['direction'] = img.GetDirection()
        return properties
    
    def get_sizes(self):
        sizes = OrderedDict()
        for img in self.image_list:
            img_nii = sitk.ReadImage(img)
            props = self.load_properties(img_nii)
            size = props['size']
            sizes[img] = size
        return sizes
    
    def get_spacings(self):
        spacings = OrderedDict()
        for img in self.image_list:
            img_nii = sitk.ReadImage(img)
            props = self.load_properties(img_nii)
            size = props['spacing']
            spacings[img] = size
        return spacings
    
    def analyze_dataset(self):
        dataset_properties = {}
        
        # different types of segmentation areas; 0 is background area
        classes = self.dataset_json['labels']
        dataset_properties['all_classes'] = [int(i) for i in classes.keys() if int(i) > 0]
        
        dataset_properties['num_cases'] = self.num_cases
        
        # sizes & spacings of each sample
        dataset_properties['all_sizes'] = self.get_sizes()
        dataset_properties['all_spacings'] = self.get_spacings()
        self.num_cases = len(self.image_list)
        dataset_properties['num_cases'] = self.num_cases
        
        # modalities
        modalities = self.dataset_json["modality"]
        dataset_properties['modalities'] = {int(k): modalities[k] for k in modalities.keys()}
        
        # TODO: intensity_properties & size_reduction
        dataset_properties['intensity_properties'] = self.collect_intensity_properties()
        # dataset_properties['size_reductions'] = self.get_size_reduction_by_cropping()  
        
        return dataset_properties
    
if __name__ == '__main__':
    path = "Task027_ACDC"
    with open(os.path.join(path, "dataset.json")) as f:
        dataset_json = json.load(f)
    image_list = get_file_list(os.path.join(path,'imagesTr'))
    image_list.sort()
    label_list = get_file_list(os.path.join(path,'labelsTr'))
    label_list.sort()
    analyzer = DatasetAnalyzer(dataset_json, image_list, label_list)
    dataset_properties = analyzer.analyze_dataset()
    print(dataset_properties.keys())
    