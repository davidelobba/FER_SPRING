import torch 
import numpy as np 
from torch.utils.data import Dataset
import pandas as pd 
from tqdm import tqdm
from random import randint
import os

class AffWild(Dataset):
    def __init__(self, samples_path=None,n_frames=25,n_mels =128,audio=False,audio_only=False,audio_separate=False,test=False, contrastive=False,twod=False, block_dimension=200, random_aug= False):
        super(AffWild, self).__init__()
        
        self.samples_path = samples_path
        self.n_frames = n_frames
        self.test = test
        self.contrastive = contrastive
        self.audio = audio
        self.random_aug = random_aug
        self.twod = twod
        self.audio_only = audio_only
        self.n_mels = n_mels 
        self.audio_separate = audio_separate 
        self.block_dimension = block_dimension
        self.data = []
        self.samples = self.split_dataset() # reference to data index and split 
        #self.data_target = [] # where the real data is stored
        #self.data_ld = [] # where the real data is stored
        #self.data_mel = [] # where the real data is stored

        
    def __len__(self):
            return len(self.samples)

    def zero_ptp(self, a):
        return 1 if np.ptp(a) == 0 else np.ptp(a)
        
    def split_dataset(self):
        # split the dataset in block of block_dimension frame
        samples = []
        for root, direct, files in os.walk(self.samples_path , topdown=False):
            for idx , f in tqdm(enumerate(files)):
                sample_data = pd.read_pickle(os.path.join(root,f))["data"]
                n_splits = int(len(sample_data)/self.block_dimension) if int(len(sample_data)/self.block_dimension) > 0 else 1
                
                sample_data = np.array(sample_data)
                #for s in sample_data:
                target = np.array([int(s["label"]) for s in sample_data])
                ld = np.array([s["ld"] for s in sample_data])
                mel_spect = np.array([s["mel_spect"] for s in sample_data])
                kx, ky =  ld[:,:,0], ld[:,:,1] 
                kx = (kx - np.min(kx))/self.zero_ptp(kx)
                ky = (ky - np.min(ky))/self.zero_ptp(ky)
                if not self.twod:
                    kz = ld[:,:,2]
                    kz = (kz - np.min(kz))/self.zero_ptp(kz)
                    ld = np.moveaxis(np.array([kx,ky,kz]),0,-1) #["len_seq", "n_kp", 3]
                else :
                    ld = np.moveaxis(np.array([kx,ky]),0,-1) #["len_seq", "n_kp", 2]
                
                for k  in range(len(sample_data)-self.block_dimension):
                    samples.append((idx,k,k+self.block_dimension))

                self.data.append((target,ld,mel_spect))
                
                # self.data_target.append(target)
                # self.data_ld.append(ld)
                # self.data_mel.append(mel_spect)
                
                #splits = np.array_split(np.array(sample_data), n_splits)
                # for sp in splits:    
                #     target = np.array([int(s["label"]) for s in sp])
                #     ld = np.array([s["ld"] for s in sp])
                #     kx, ky =  ld[:,:,0], ld[:,:,1] 
                #     kx = (kx - np.min(kx))/self.zero_ptp(kx)
                #     ky = (ky - np.min(ky))/self.zero_ptp(ky)
                #     if not self.twod:
                #         kz = ld[:,:,2]
                #         kz = (kz - np.min(kz))/self.zero_ptp(kz)
                #         ld = np.moveaxis(np.array([kx,ky,kz]),0,-1) #["len_seq", "n_kp", 3/2]
                #     else :
                #         ld = np.moveaxis(np.array([kx,ky]),0,-1) #["len_seq", "n_kp", 3/2]
                        
                #     mel_spect = np.array([s["mel_spect"] for s in sp])
                #     samples.append((target, ld, mel_spect))
                
        return samples
    
    def __getitem__(self, index: int):
        idx, from_block, to_block = self.samples[index]
        target, ld, mel_spect = self.data[idx]
        target, ld, mel_spect = target[from_block:to_block], ld[from_block:to_block], mel_spect[from_block:to_block]
        #target, ld, mel_spect = self.samples[index][0], self.samples[index][1], self.samples[index][2]
        if self.random_aug:
            ld =  ld + np.random.normal(0, 0.005, size=(ld.shape[1], ld.shape[2]))
        
        if not self.audio_separate:
            mel_spect = mel_spect.mean(1)
            mel_spect = np.array([np.repeat(mv,68) for mv in mel_spect])
            if self.twod and self.audio:
                ld = torch.Tensor(np.moveaxis(np.array([ld[:,:,0],ld[:,:,1],mel_spect]),0,-1))
            elif self.audio:
                ld = torch.Tensor(np.moveaxis(np.array([ld[:,:,0],ld[:,:,1],ld[:,:,2],mel_spect]),0,-1))
            else:
                mel_spect = torch.Tensor(mel_spect)
                ld = torch.Tensor(ld)        
        else:
            mel_spect = torch.Tensor(mel_spect)
            ld = torch.Tensor(ld)    
        
        num_frames = ld.shape[0]-1
        if num_frames > self.n_frames:
            start_frame =  randint(0, num_frames-self.n_frames)
        else:
            start_frame = 0
        
        if num_frames < self.n_frames:
            pad = self.n_frames - num_frames
            #ld = torch.cat((ld,ld[ld.shape[0]-1].repeat(pad,1,1)), 0)
            ld = torch.cat((ld,ld[:pad]), 0)
            target = torch.Tensor(target)
            target = torch.cat((target,target[:pad]), 0).numpy()
            if self.audio and self.audio_separate:
                mel_spect =torch.cat((mel_spect,mel_spect[:pad]), 0)
        
        if self.test:
            start_frame = 0 
        
        last_frame = start_frame + self.n_frames
        
        if self.contrastive:
            if num_frames > self.n_frames:
                start_frame_2 =  randint(0, num_frames-self.n_frames)
            else:
                start_frame_2 = 0
            start_frame_2 = 0 if self.test else start_frame_2

            if self.audio_only:
                return target[last_frame], mel_spect[start_frame: start_frame+self.n_frames,:], mel_spect[start_frame_2: start_frame_2+self.n_frames,:] #[num_f, n_mel]
            elif self.audio_separate:
                return target[last_frame], ld[start_frame: start_frame+self.n_frames,17:,: ], ld[start_frame_2: start_frame_2+self.n_frames,17:,: ], mel_spect[start_frame: start_frame+self.n_frames,:], mel_spect[start_frame_2: start_frame_2+self.n_frames,:]
            else:
                return target[last_frame], ld[start_frame: start_frame+self.n_frames,17:,: ], ld[start_frame_2: start_frame_2+self.n_frames,17:,: ]    
        else:
            if self.audio_only:
                return target[last_frame], mel_spect[start_frame: start_frame+self.n_frames,:]
            elif self.audio_separate:
                return target[last_frame], ld[start_frame: start_frame+self.n_frames,17:,: ], mel_spect[start_frame: start_frame+self.n_frames,:]
            else:
                return target[last_frame], ld[start_frame: start_frame+self.n_frames,17:,: ]
