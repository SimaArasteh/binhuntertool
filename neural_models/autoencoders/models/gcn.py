import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, layer_dims, nout, dropout, softmax, name):
        super(GCN, self).__init__()
        self.layers = []
        self.num_layers = len(layer_dims) + 1
        self.build_model(nfeat, layer_dims, nout)
        self.softmax = softmax
        self.name = name
        print(self)

    def build_model(self, nfeat, layer_dims, nout):
        self.add_gcn_layer(GraphConvolution, nfeat, layer_dims[0], i=1)
        for i in range(1, len(layer_dims)):
            self.add_gcn_layer(GraphConvolution, layer_dims[i-1], layer_dims[i], i+1)
        self.add_gcn_layer(GraphConvolution, layer_dims[-1], nout, self.num_layers)

    def add_gcn_layer(self, layer, in_dim, out_dim, i):
        new_layer = layer(in_dim, out_dim)
        self.layers.append(new_layer)
        self.add_module(module=new_layer, name="gc{}".format(i))
        
    def forward(self, x, adj):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x, adj))
        x = self.layers[-1](x, adj)
        if self.softmax:
            x = F.log_softmax(x, dim=1)
        return x