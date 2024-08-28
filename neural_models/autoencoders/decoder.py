from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import OrderedDict
import math
import numpy as np
import time

# class Decoder(object):
#     def __init__(self, placeholders, input_dim, **kwargs):
#         super(Decoder, self).__init__(**kwargs)

#         self.nodes = placeholders['l_nodes'] # labels
#         self.edges = placeholders['l_edges'] # labels
#         self.z = placeholders['z'] # latent representation of the graph
#         self.num_node_type = self.nodes.shape[0]
#         self.placeholders = placeholders
        
# #         self.num_node_reconstruction_layers = 2
# #         self.num_edge_reconstruction_layers = 2

#         self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

#         self.build()
        
#     def build(self):
#         self.node_reconstruction_layers = []
#         self.node_reconstruction_layers.append(layers.Dense(input_dim=self.z.shape[0], 
#                                                             output_dim=1000, 
#                                                             placeholders=self.placeholders, 
#                                                             act=tf.relu))
#         self.node_reconstruction_layers.append(layers.Dense(input_dim=1000, 
#                                                             output_dim=self.z.shape[0], 
#                                                             placeholders=self.placeholders,
#                                                             act=lambda x:x))
        
#         self.edge_reconstruction_layers = []
        
        
    
#     def node_reconstruction(self, z):
#     layers.Dense(input_dim=self.gcn_out_dim, 
#                                        output_dim=self.output_dim, 
#                                        placeholders=self.placeholders,
#                                        act=lambda x: x, 
#                                        dropout=True)
#     pass

#     def edge_reconstruction(self, r_nodes, nodes):
#     pass


class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, name, has_input=True, has_output=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.name = name

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw

#     def register_backward_hook(self, hook):
#         print ('Going backward in GRU ', self.name)
#         return
