import torch
from torchvision.datasets.vision import VisionDataset
import torchvision
import os
import random
import cv2
import numpy as np
import json
from random import randint
from PIL import Image,  ImageSequence


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

class CAER(VisionDataset):
    def __init__(self, root, transform = True, dim_image=64, min_frames=25):
        super(CAER, self).__init__(root,transform=transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.dim_image = dim_image
        self.min_frames = min_frames

        self.resize = torchvision.transforms.Resize(self.dim_image)
        self.flip_h = torchvision.transforms.RandomHorizontalFlip(1) #1 probability 
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage() 

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
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample1, sample2 = self.video_loader(path)
        return sample1, sample2, target
    
    def __len__(self):
        return len(self.samples)

    def video_loader(self, path):
        # TODO implement also other augmentation strategies
        # right now only flip and time shift

        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))        
        sequence = torch.Tensor()

        while True:
            succesful, frame = cap.read()
            if succesful:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.resize(frame)
                frame = self.to_tensor(frame) # [c,h,w]
                sequence = torch.cat((sequence,frame.unsqueeze(0)), 0)
            else:
                break 
        cap.release()

        # if the sequence is smaller than the minimum pad use the last frame to pad
        if num_frames < self.min_frames:
            pad = self.min_frames - num_frames
            sequence = torch.cat((sequence,sequence[sequence.shape[0]-1].unsqueeze(0).repeat(pad, 1, 1, 1)),0)
        
        # flip the all sequence
        sequence_flip = self.flip_h(sequence)

        #because we are considering chunk of self.min_frames sequences 
        start_frame =  randint(0, num_frames-self.min_frames)
        start_frame_flip =  randint(0, num_frames-self.min_frames)

        
        return sequence[start_frame:start_frame+self.min_frames], sequence_flip[start_frame_flip:start_frame_flip+self.min_frames]

        
        

