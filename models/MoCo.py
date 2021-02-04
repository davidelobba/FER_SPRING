import torch


class MoCo(torch.nn.Module):
    def __init__(self, base_encoder, dim=128,m=0.999, K=512):
        """
        dim: feature dimension (default: 128)
        K: queue size;
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m

        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()
        

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_features1", torch.randn(K,1,dim))
        self.queue_features1= torch.nn.functional.normalize(self.queue_features1, dim=2)
        self.register_buffer("queue_features2", torch.randn(K,1,dim))
        self.queue_features2= torch.nn.functional.normalize(self.queue_features2, dim=2)
        self.register_buffer("queue_ptr_features", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_targets", torch.randint(0,7,(K,)))
        self.register_buffer("queue_ptr_targets", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self,features1, features2, targets):

        batch_size = features1.shape[0]

        ptr_feat = int(self.queue_ptr_features)
        ptr_tar = int(self.queue_ptr_targets)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_features1[ptr_feat:ptr_feat + batch_size,:,:] = features1
        self.queue_features2[ptr_feat:ptr_feat + batch_size,:,:] = features2
        ptr_feat = (ptr_feat + batch_size) % self.K  # move pointer
        self.queue_targets[ptr_tar:ptr_tar + batch_size] = targets
        ptr_tar = (ptr_tar + batch_size) % self.K  # move pointer

        self.queue_ptr_features[0] = ptr_feat
        self.queue_ptr_targets[0] = ptr_tar


    def forward(self, input_q, input_k, targets,train=True):
        """
        Input:
            inputs: [batch_size*seq_length*2, 3, 224, 224]
            targets: [batch_size]
        Output:
            features: [K + batch_size*2,128]
            labels: [K + batch_size*2]
            video_features: [batch_size*2,2048]
        """
        #batch_size = targets.size(0)
        #input_q, input_k = torch.split(inputs, [batch_size*final_frames, batch_size*final_frames], dim=0)

        q1, video_feat_q1 = self.encoder_q(input_q)
        q2, video_feat_q2 = self.encoder_q(input_k)
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)

        with torch.no_grad():  # no gradient to keys
          self._momentum_update_key_encoder()  # update the key encoder
          k1, video_feat_k = self.encoder_k(input_q)  # keys: NxC
          k2, video_feat_k = self.encoder_k(input_k)
          k1 = k1.unsqueeze(1)
          k2 = k2.unsqueeze(1)

        features = torch.cat([q1, q2, k1.detach(), k2.detach(), self.queue_features1.clone().detach(), self.queue_features2.clone().detach()])
        labels = torch.cat([targets, targets, targets, targets,self.queue_targets, self.queue_targets])

        video_features = torch.cat([video_feat_q1, video_feat_q2], dim = 0)

        if train == True:
          self._dequeue_and_enqueue(k1, k2, targets)

        return features, labels, video_features