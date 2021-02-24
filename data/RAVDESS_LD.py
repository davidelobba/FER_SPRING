import torch
from torch.utils.data import Dataset
import os
import json
from random import randint
from torch import nn
from torch.nn import functional as F
import pandas as pd 
import numpy as np
import librosa
from tqdm import tqdm

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

class RAVDESS_LANDMARK(Dataset):
    def __init__(self, root, samples=None,min_frames=25,audio=False, test=False, zero_start=False, contrastive=False, mixmatch=False, random_aug=False, drop_kp=False):
        super(RAVDESS_LANDMARK, self).__init__()
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        if samples is None:
            samples = make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.min_frames = min_frames
        self.test = test
        self.zero_start = zero_start
        self.contrastive = contrastive
        self.mixmatch = mixmatch
        self.random_aug = random_aug
        self.drop_kp = drop_kp
        self.audio = audio
        #if self.audio:
        #    self.preprocess_audio()

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
    
    def preprocess_audio(self):
        new_samples = []
        for idx in tqdm(range(len(self.samples))):
            path_ld = self.samples[idx][0]
            data = pd.read_csv(path_ld)
            kx = data.iloc[:,297:365].to_numpy()
            path_audio = self.samples[idx][2]
            data, sampling_rate = librosa.load(path_audio)
            #mel_spect = np.array(librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_mels=kx.shape[0])).mean(axis=1)
            
            hop_length = int(len(data)/kx.shape[0])
            mel_spect = np.array(librosa.feature.melspectrogram(y=data, sr=sampling_rate, hop_length=hop_length, n_mels=128)) #.mean(0)
            mel_spect = mel_spect[:kx.shape[0]]
            new_samples.append( (self.samples[idx][0],self.samples[idx][1],self.samples[idx][2],  mel_spect) ) 

        self.samples = new_samples
    
    def get_class_sample_count(self):
        count =np.zeros(len(self.classes), dtype=int)
        for s in self.samples:
            count[s[1]] +=1
        weight = 1. /count
        sample_weight = []
        for s in self.samples:
            sample_weight.append(weight[s[1]]) 
        return count, torch.Tensor(sample_weight)
    
    def rotate(self, out, origin=(0.5, 0.5), degrees=0):
        out_rot = torch.Tensor([])
        for p in out:
            angle = np.deg2rad(degrees)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]])
            o = np.atleast_2d(origin)
            p = np.atleast_2d(p)
            p = np.squeeze((R @ (p.T-o.T) + o.T).T)
            out_rot = torch.cat((out_rot,torch.Tensor(p).unsqueeze(0)))
        return out_rot

    def __getitem__(self, index: int):
        path, target  = self.samples[index][0], self.samples[index][1]
        data = pd.read_csv(path)

        noise = torch.normal(0, 0.003, size=(51, 2))
        
        kx = data.iloc[:,297:365].to_numpy()
        ky = data.iloc[:,365:433].to_numpy()
        
        kx = (kx - np.min(kx))/np.ptp(kx)
        ky = (ky - np.min(ky))/np.ptp(ky)    
        
        if self.audio:
            path_audio = self.samples[index][2]
            data, sampling_rate = librosa.load(path_audio)
            # mel_spect = self.samples[index][3]
            # hop_length = int(len(data)/kx.shape[0])
            # mel_spect = np.array(librosa.feature.melspectrogram(y=data, sr=sampling_rate, hop_length=hop_length, n_mels=128)).mean(0)
            mel_spect = mel_spect[:kx.shape[0]]
            mel_spect = np.array([np.repeat(mv,68) for mv in mel_spect])
            ld = np.array([kx,ky,mel_spect]).T
        else:
            ld = np.array([kx,ky]).T

        ld = torch.Tensor(np.rollaxis(ld, 1, 0))
        num_frames = ld.shape[0] 
        
        if num_frames < self.min_frames:
            pad = self.min_frames - num_frames
            #ld = torch.cat((ld,ld[ld.shape[0]-1].repeat(pad,1,1)), 0)
            ld = torch.cat((ld,ld[:pad]), 0)
                
        
        if num_frames > self.min_frames:
            start_frame =  randint(0, num_frames-self.min_frames)
        else:
            start_frame = 0
        
        if self.test:
            start_frame = 0 

        if self.zero_start:
            start_frame = 0
        
        if self.contrastive:
            # in case of contrastive return two version sampled in different situation
            if num_frames > self.min_frames:
                start_frame_2 =  randint(0, num_frames-self.min_frames)
            else:
                start_frame_2 = 0
            return target, ld[start_frame: start_frame+self.min_frames,17:,: ], ld[start_frame_2: start_frame_2+self.min_frames,17:,: ]
        else:
            if self.random_aug and not self.test:
                ld =  ld[start_frame: start_frame+self.min_frames,17:,: ]
                ld_vflip  = ld.clone()
                ld_vflip[:,:,0] = 1 - ld_vflip[:,:,0]
                ld_noise = ld.clone() + noise
                out = torch.cat((ld.unsqueeze(0), ld_vflip.unsqueeze(0), ld_noise.unsqueeze(0)),0)
                idx = randint(0,2)
                out = out[idx]
                if self.drop_kp and bool(randint(0,1)):
                    ## randomly put to zero 10% of the landmark aka 5 points 
                    rnd = torch.rand(self.min_frames,5)*51   
                    kps = rnd.int()
                    kps = kps.numpy()
                    for idx in range(out.shape[0]):
                        out[idx, kps[idx],:] = 0 
                out = self.rotate(out, degrees=randint(-10,10))
                return target, out  

            return target, ld[start_frame: start_frame+self.min_frames,17:,: ]