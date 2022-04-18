import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from fd.layers import ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d, CBatchNorm1d_legacy, ResnetBlockConv1d

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

class DGCNN_cls(nn.Module):
    def __init__(self, k, emb_dims):
        super(DGCNN_cls, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512,  self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear4 = nn.Linear(self.emb_dims*2, 512)
        
     #   self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
      #  self.bn6 = nn.BatchNorm1d(512)
      #  self.dp1 = nn.Dropout(p=args.dropout)
       # self.linear2 = nn.Linear(512, 256)
       # self.bn7 = nn.BatchNorm1d(256)
       # self.dp2 = nn.Dropout(p=args.dropout)
      #  self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):

        x= x.reshape(x.shape[0]*x.shape[1], x.shape[2],x.shape[3])

        batch_size = x.size(0)
        x=x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)


        return x


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

    feature = x.view(batch_size*num_points, -1)[idx, :]

    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)



def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Decoder(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=128, leaky=False):
        super().__init__()

        self.c_dim = c_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)

        self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size*2)
        self.block1 = ResnetBlockFC(hidden_size*2)
        self.block2 = ResnetBlockFC(hidden_size*2)
        self.block3 = ResnetBlockFC(hidden_size*2)
        self.block4 = ResnetBlockFC(hidden_size*2)

        self.fc_out = nn.Linear(hidden_size*2, 3)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c):
        batch_size, T, D = p.size()
        #print(p.shape, c.shape)

        net = self.fc_p(p)



        net_c = self.fc_c(c)

        net_c =net_c.unsqueeze(1).expand(net.shape)


        net = torch.cat((net,net_c),2)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)
        #print(net.shape)
        out = self.fc_out(self.actvn(net))

        return out


class pyramid_Decoder3(nn.Module):

    
    def __init__(self):
        super().__init__()

        # Submodules

        self.fc_c = nn.Linear(2048, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.block1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.block2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc_out  = nn.Linear(128, 1)
        self.bn3=nn.BatchNorm1d(1)

    def forward(self, c):

        x=F.relu(self.bn0(self.fc_c(c))) # bnumber,2048+64 -> # bnumber,1024 
        x=F.relu(self.bn1(self.block1(x)))
        x=F.relu(self.bn2(self.block2(x)))  # bnumber,512 -> # bnumber,128 
        x=self.bn3(self.fc_out(x)).squeeze()

        return x

    

class OccupancyNetwork(nn.Module):

    def __init__(self, decoder, encoder, device):
        super().__init__()
        decoder=nn.DataParallel(decoder)
        encoder=nn.DataParallel(encoder)
        self.decoder = decoder.to(device)
        self.encoder = encoder.to(device)
        self._device = device


    def forward(self, inputs):

        c = self.encode_inputs(inputs)
        n = self.decode(c)
        return n

    def compute_loss(self, inputs, output):

        c = self.encode_inputs(inputs)
        n1 = self.decode(c)
        output=output.reshape(-1)
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(n1, output)

        return loss

    def encode_inputs(self, inputs):
        c = self.encoder(inputs)
        return c

    def decode(self, c):
        n = self.decoder(c)
        return n
    
