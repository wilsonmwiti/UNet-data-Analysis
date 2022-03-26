import os
import pickle
import glob
import json
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk
from multiprocessing import Pool


class IntensityAnalyzer():
    def __init__(self, dataset_json, image_list, label_list):
        self.dataset_json = dataset_json
        self.image_list = image_list
        self.label_list = label_list
        self.num_cases = len(self.image_list)
        
    @staticmethod
    def _compute_stats(voxels):
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        std = np.std(voxels)
        min = np.min(voxels)
        max = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, std, min, max, percentile_99_5, percentile_00_5
    
    def _get_voxels_in_foreground(self, image, label):
        mask = label > 0
        voxels = list(image[mask][::10])
        return voxels
    
    def _get_array_from_nii(self, nii_file):
        array = sitk.ReadImage(nii_file)
        array = sitk.GetArrayFromImage(array)
        return array
    
    def collect_intensity_properties(self):
        results = OrderedDict()
        props_per_case = OrderedDict()
        all_voxels = []
        for image, label in zip(self.image_list, self.label_list):
            local_props = OrderedDict()
            image_array = self._get_array_from_nii(image)
            label_array = self._get_array_from_nii(label)
            case_voxels = self._get_voxels_in_foreground(image_array, label_array)
            median, mean, std, min, max, percentile_99_5, percentile_00_5 = self._compute_stats(case_voxels)
            local_props['median'] = median
            local_props['mean'] = mean
            local_props['std'] = std
            local_props['min'] = min
            local_props['max'] = max
            local_props['percentile_99_5'] = percentile_99_5
            local_props['percentile_00_5'] = percentile_00_5
            props_per_case[image] = local_props
            all_voxels += case_voxels
        median, mean, std, min, max, percentile_99_5, percentile_00_5 = self._compute_stats(all_voxels)
        results['median'] = median
        results['mean'] = mean
        results['std'] = std
        results['min'] = min
        results['max'] = max
        results['percentile_99_5'] = percentile_99_5
        results['percentile_00_5'] = percentile_00_5
        results['props_per_case'] = props_per_case
        return results

        
