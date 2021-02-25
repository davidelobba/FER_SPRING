import torchvision
import torch
from models.STGCN import STGCN
from models.TCN import TCN
import torch.nn as nn
import yaml
import numpy as np
from models.STGCN import get_normalized_adj


class Encoder(torch.nn.Module):
    def __init__(self, num_nodes_ld=51,num_nodes_audio=64,num_feat_video=2,num_feat_audio=1,config_file=None, num_classes=128, device="cuda:0"):
        """
        """
        super(Encoder, self).__init__()

        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        with open(config["model_params"]["adj_matr"], 'rb') as f:
            A = np.load(f)
        self.A_hat_v = torch.Tensor(get_normalized_adj(A)).to(device)
        adj_a  = np.ones((num_nodes_audio,num_nodes_audio))- np.identity(num_nodes_audio)
        self.A_hat_a = torch.Tensor(get_normalized_adj(adj_a)).to(device)

        self.video_stgcn = STGCN(num_nodes_ld,num_feat_video,config["dataset"]["min_frames"],config["model_params"]["feat_out"], num_classes=num_classes,edge_weight=config["model_params"]["edge_weight"], contrastive=False)
        #self.audio_stgcn = STGCN(num_nodes_audio,num_feat_audio,config["dataset"]["min_frames"],config["model_params"]["feat_out"], num_classes=num_classes,edge_weight=config["model_params"]["edge_weight"], contrastive=False)
        self.audio = TCN(in_chan=128, n_blocks=5, n_repeats=2, out_chan=num_classes) #AudioEncoder(out=num_classes)
        #model = model.to(args.device)
        self.fc_mix = nn.Linear(num_classes*2,512)
        self.fc_out = torch.nn.Sequential(torch.nn.Linear(512, 256),torch.nn.ReLU(),torch.nn.Linear(256, num_classes))


    def forward(self, inputs_video, inputs_audio):
        """
        Input:
            inputs: [batch_size * seq_length, 3, 224, 224]
        Output:
            outputs: [batch_size, 128]
            video_features: [batch_size,2048]
        """
        vf_q1 = self.video_stgcn(self.A_hat_v, inputs_video)
        #print(vf_q1.shape)
        #print(self.A_hat_a.shape)
        #print(inputs_audio.shape)

        af_q2 = self.audio(inputs_audio)
        #print(af_q2.shape)

        feat = torch.cat((vf_q1,af_q2),1)
        #print(feat.shape)
        feat = self.fc_mix(feat)
        feat_out = self.fc_out(feat)
        #print(feat.shape)
        #print(feat_out.shape)

        return torch.nn.functional.normalize(feat_out), feat
