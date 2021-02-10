from dgl.nn.pytorch import GATConv
from dgl import DGLGraph
import dgl
import json
import torch 
from random import randint
from torch import nn
from torch.nn import functional as F

from scipy.spatial import distance

# implementation from dgl website
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        # self.attn_fc = nn.Linear( out_dim, 1, bias=False)

        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()

        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, feat):
        out = torch.Tensor([]).to(feat.device)
        for batch_idx in range(feat.shape[0]):
            h = self.layer1(feat[batch_idx])
            h = F.elu(h)
            h = self.layer2(h).mean(1)
            out = torch.cat((out, h.unsqueeze(0)), 0)

        return out


class FER_GAT(nn.Module):
    def __init__(self, out_gat: int = 128, num_of_ld=51, in_dim=128, hidden_dim=64, 
                 memory_attended=True, att_head=4, lstm_dim = 32, n_classes=8,num_frames=50,  device="cpu"):
        
        super().__init__()

        # create the base graph structure
        G = dgl.DGLGraph()
        G.add_nodes(num_of_ld)
        start = []
        end = []
        for i in range(0, num_of_ld):
            for k in range(0, num_of_ld):
                if i != k:
                    start.append(i)
                    end.append(k)
                    
        G.add_edges(start, end)
        G = G.to(device)
        
        self.gat = GAT(G,
                       in_dim=2,
                       hidden_dim=hidden_dim,
                       out_dim=out_gat,
                       num_heads=att_head)
        
        self.lstm_encoder = nn.LSTM(num_of_ld, lstm_dim, 2, batch_first = True, dropout=0.1)
        self.fc = nn.Linear(num_frames*lstm_dim, n_classes)


    def forward(self, features) -> torch.Tensor:
        
        # loop trough landmark dimension
        extracted = torch.tensor([]).to(features.device)
        
#         for idx in range(features.shape[2]):
#             out , (h,c) = self.lstm_encoder(features[:,:,idx,:].permute(1,0,2))            
#             extracted = torch.cat((extracted, out.permute(1,0,2)[:,-1,:].unsqueeze(1)),1)

        for batch_idx in range(features.shape[1]): # loop over num_frames dimension
            feat = self.gat(features[:,batch_idx])
            extracted = torch.cat((extracted, feat.unsqueeze(1)),1)
            
        # [seq_len, batch, input_size]
        out,_ = self.lstm_encoder(extracted.permute(1,0,2))
        out = out.permute(1,0,2)
        out = self.fc(out.flatten(start_dim=1))
        return out