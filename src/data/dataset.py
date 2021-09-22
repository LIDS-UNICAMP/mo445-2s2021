# Dataset
import os
from os import path
from torch.utils.data import Dataset
from flim.experiments import utils
import numpy as np
from PIL import Image
import torch
import cv2


class SegmDataset(Dataset):
    def __init__(self, input_dir,  gts_dir, transform=None):
        assert isinstance(input_dir, str) and len(input_dir) > 0,\
            "Invalid input_dir"
            
        assert isinstance(gts_dir, str) and len(gts_dir) > 0,\
            "Invalid gts_dir"

        self._gts_dir = gts_dir
        self._in_dir  = input_dir
        
        self._transform = transform

        self._image_names = None
        self._gt_names = None

        self._load_dataset_info()

    def __getitem__(self, index):

        gts_name   = self._gt_names[index]
        label_path = os.path.join(self._gts_dir, f"{gts_name}_label.png")
        image_path = os.path.join(self._in_dir, f"{gts_name}.png")

        image = utils.load_image(image_path)
        
        label_image = self._load_label_image(label_path)

        if label_image.max() == 255:
            label_image[label_image == 255] = 1
        else:
            maxv = label_image.max()
            mask = label_image==maxv
            label_image[mask]=1
            label_image[np.invert(mask)]=0

        if(self._transform):
            image = self._transform(image)
        sample = (image, label_image.astype(np.int64))
        
        return sample 

    def _load_dataset_info(self):
     
        if path.exists(self._in_dir):
            self._image_names = []
            for name in os.listdir(self._in_dir):
                if name.endswith('.png'):
                    self._image_names.append(name.split('.png')[0])
                else:
                    continue
            self._image_names.sort()
        else:
            raise ValueError(f"{self._in_dir} does not exists")

        if path.exists(self._gts_dir):
            self._gt_names=[]
            for name in os.listdir(self._gts_dir):
                if name.endswith('_label.png'):
                    self._gt_names.append(name.split('_label')[0])
                else:
                    continue
            self._gt_names.sort()
        else:
            raise ValueError(f"{self._gts_dir} does not exists")

        isok=True
        for el in self._gt_names:
            if not (el in self._gt_names):
                isok=False
                raise ValueError(f"Missing {el} from images")
        

    def __len__(self):
        return len(self._gt_names)

    def _load_label_image(self, label_path):
        label_image = np.array(Image.open(label_path))

        return label_image



        
class ToTensor(object):
    def __call__(self, sample):
        image = np.array(sample)
        image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image.copy()).float()