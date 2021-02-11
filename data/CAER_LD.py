import torch
from torch.utils.data import Dataset
import os
import json
from random import randint
from torch import nn
from torch.nn import functional as F
import pandas as pd 
import numpy as np

def make_dataset(directory, class_to_idx):
    instances = []
    directory = os.path.expanduser(directory)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
    return instances

class CAER_LANDMARK(Dataset):
    def __init__(self, root, min_frames=25, test=False):
        super(CAER_LANDMARK, self).__init__()
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.min_frames = min_frames
        self.test = test
        

    def __len__(self):
            return len(self.samples)


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        

    def __getitem__(self, index: int):
        path, target  = self.samples[index][0], self.samples[index][1]

        with open(path) as json_file:
            data = json.load(json_file)
        ## for the time being just slice 
        ld = np.array(data["landmark"])

        #pad if the are not enough frames
        kx = ld[:,:,0]
        ky = ld[:,:,1]
        kx = (kx - np.min(kx))/np.ptp(kx)
        ky = (ky - np.min(ky))/np.ptp(ky)
        norm_ld = np.array([kx,ky]).T
        ld = torch.Tensor(np.rollaxis(norm_ld,1,0))

        num_frames = ld.shape[0] 
        
        if num_frames < self.min_frames:
            pad = self.min_frames - num_frames
            ld = torch.cat((ld,ld[ld.shape[0]-1].repeat(pad,1,1), 0))

        if num_frames > self.min_frames:
            start_frame =  randint(0, num_frames-self.min_frames)
        else:
            start_frame = 0 

        if self.test:
            start_frame =num_frames - self.min_frames
    
        return target, ld[start_frame: start_frame+self.min_frames,17:,: ]