import torch.nn as nn
import torch.nn.functional as F

from .model_utils import create_layers
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, nfeat, layer_dims, nout, name):
        super(FC, self).__init__()
        self.layers = []
        self.num_layers = len(layer_dims) + 1
        self.build_model(nfeat, layer_dims, nout)
        self.name = name
        print(self)

    def build_model(self, nfeat, layer_dims, nout):
        self.add_layer(nn.Linear, nfeat, layer_dims[0], i=1)
        for i in range(1, len(layer_dims)):
            self.add_layer(nn.Linear, layer_dims[i - 1], layer_dims[i], i+1)
        self.add_layer(nn.Linear, layer_dims[-1], nout, self.num_layers)

    def add_layer(self, layer, in_dim, out_dim, i):
        new_layer = layer(in_dim, out_dim)
        self.layers.append(new_layer)
        self.add_module(module=new_layer, name="fc{}".format(i))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
