import torchvision
import torch
from models.STGCN import STGCN
from models.TCN import TCN
import torch.nn as nn
import yaml
import numpy as np
from models.STGCN import get_normalized_adj


class Encoder(torch.nn.Module):
    def __init__(self, num_nodes_ld=51,num_nodes_audio=64,num_feat_video=2,num_feat_audio=1,config_file=None, num_classes=256, device="cuda:0"):
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
        self.fc_mix_1 = nn.Linear(num_classes*2,256)
        self.fc_mix_2 = nn.Linear(256+num_classes,512)
        
        self.fc_out = torch.nn.Sequential(torch.nn.Linear(512+num_classes, 256),torch.nn.ReLU(),torch.nn.Linear(256, num_classes*2))


    def forward(self, inputs_video, inputs_audio):
        """
        Input:
            inputs_video: [batch, len_seq, n_nodes, n_feat]
            inputs_video: [batch, len_seq, n_mels]
        Output:
            outputs: [batch_size, 128]
            video_features: [batch_size,2048]
        """
        vf_q1 = self.video_stgcn(self.A_hat_v, inputs_video)
        af_q2 = self.audio(inputs_audio)
        #ff = (vf_q1*af_q2)#.sum(-1) #torch.dot(af_q2,vf_q1)
        #print(ff.shape)
        

        feat = torch.cat((vf_q1,af_q2),1)
        #print(feat.shape)
        feat = self.fc_mix_1(feat)
        feat = torch.cat((feat,vf_q1),1)
        feat = self.fc_mix_2(feat)
        feat_tm = torch.cat((feat,vf_q1),1)

        feat_out = self.fc_out(feat_tm)
        #print(feat.shape)
        #print(feat_out.shape)
        #print(feat.shape)
        #print(feat_out.shape)


        return torch.nn.functional.normalize(feat_out), feat
