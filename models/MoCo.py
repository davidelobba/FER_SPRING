import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,  encoder, dim=8, K=512, m=0.999, T=0.07, num_sample = 32, mlp=False, A = None, config=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        
        # number of sample to extract when using the queue
        self.num_sample = num_sample 

        # create the encoders
        # num_classes is the output fc dimension

        self.encoder_q = encoder(51, 2, config["dataset"]["train"]["min_frames"],config["model_params"]["feat_out"], num_classes=config["dataset"]["train"]["classes"], edge_weight=config["model_params"]["edge_weight"], contrastive=config["training"]["contrastive"]) 
        self.encoder_k = encoder(51, 2, config["dataset"]["train"]["min_frames"],config["model_params"]["feat_out"], num_classes=config["dataset"]["train"]["classes"], edge_weight=config["model_params"]["edge_weight"], contrastive=config["training"]["contrastive"])

        self.A = A

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_feat", torch.randn(K,4,dim))
        self.queue_feat = torch.nn.functional.normalize(self.queue_feat, dim=2)
        self.register_buffer("queue_tgt", torch.zeros(K, dtype=torch.float))
        self.queue_tgt = torch.nn.functional.normalize(self.queue_tgt, dim=0)
        

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, features,targets):
        # gather keys before updating queue

        batch_size = features.shape[0]
        ptr = int(self.queue_feat.shape[0]) 
        
        self.queue_feat = torch.cat((self.queue_feat, features),0)
        self.queue_feat = self.queue_feat[batch_size:ptr + batch_size, :, :] 

        self.queue_tgt = torch.cat((self.queue_tgt,targets),0)
        self.queue_tgt = self.queue_tgt[batch_size:ptr + batch_size] 


    def forward(self, input_1, input_2,targets, train=False):
        """
        Input:
            input_q: [batch, len_seq, num_nodes, feat]
            input_k: [batch, len_seq, num_nodes, feat]
            target: [batch]
        Output:
            logits, targets
        """


        # compute query features
        q1, vf_q1 = self.encoder_q(self.A, input_1)
        q1 = q1.unsqueeze(1)  # [batch,num_feat]
        q2, vf_q2 = self.encoder_q(self.A, input_1)
        q2 = q2.unsqueeze(1)  # [batch,num_feat]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k1, vf_k1 = self.encoder_k(self.A,input_1)
            k1 = k1.unsqueeze(1)  # keys: NxC
            k2, vf_k2 = self.encoder_k(self.A,input_2)
            k2 = k2.unsqueeze(1)  # keys: NxC

        # compute logits
        features = torch.cat([q1, q2], dim=1) #, k1.detach(), k2.detach()
        
        if train:
            video_features = torch.cat([vf_q1, vf_q1], dim = 0)
        else:
            video_features = vf_q1

        #if train:
        #    self._dequeue_and_enqueue(features,targets)
        
        #features = torch.cat((features,self.queue_feat[:self.num_sample ,:,:].clone().detach()))
        #targets = torch.cat((targets,self.queue_tgt[:self.num_sample].clone().detach()))

        return features, targets, video_features
