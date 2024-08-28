import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle as pkl
import scipy.sparse as sp
import shutil
import time
import torch
import torchvision as tv
import torch.nn as nn
import random

from torch.autograd import Variable
from random import shuffle
# from model import *
# from utils import *


def encode_adj(adj, max_prev_node=10, is_full = False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output

def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

def test_encode_decode_adj():
######## code test ###########
    G = nx.ladder_graph(5)
    G = nx.grid_2d_graph(20,20)
    G = nx.ladder_graph(200)
    G = nx.karate_club_graph()
    G = nx.connected_caveman_graph(2,3)
    print(G.number_of_nodes())
    
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    #
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    print('adj\n',adj)
    adj_output = encode_adj(adj,max_prev_node=5)
    print('adj_output\n',adj_output)
    adj_recover = decode_adj(adj_output,max_prev_node=5)
    print('adj_recover\n',adj_recover)
    print('error\n',np.amin(adj_recover-adj),np.amax(adj_recover-adj))
    
    
    adj_output = encode_adj_flexible(adj)
    for i in range(len(adj_output)):
        print(len(adj_output[i]))
    adj_recover = decode_adj_flexible(adj_output)
    print(adj_recover)
    print(np.amin(adj_recover-adj),np.amax(adj_recover-adj))

# def topologicalSortUtil(graph,node_idx,v,visited,stack): 
#     # Mark the current node as visited. 
#     visited[v] = True
  
#     # Recur for all the vertices adjacent to this vertex 
#     for i in graph.successors(v): 
#         if visited[i] == False: 
#             topologicalSortUtil(graph,node_idx,i,visited,stack) 
  
#     # Push current vertex to stack which stores result 
#     stack.insert(0,node_idx[v])
    
# def topologicalSort(graph): 
#     # Mark all the vertices as not visited 
#     visited = {n:False for n in graph.nodes()}
#     stack = [] 

#     node_idx = {n:i for i, n in enumerate(graph.nodes())}
#     for i in graph: 
#         if visited[i] == False: 
#             topologicalSortUtil(graph,node_idx,i,visited,stack) 
#     # Print contents of the stack 
#     return stack 

def traverse_graph(adj, mask_size):
    queue = [np.random.randint(adj.shape[0])]
    masked_idx = [queue[0]]
    visited = set()
    while len(queue) > 0:
        node = queue[0]
        queue = queue[1:]
        successors = np.where(adj[node])[0]
        for child in successors:
            mask_node = True
            # check the node is not already marked for masking
            if child in masked_idx:
                continue
            if child in visited:
                # loop encountered, try again
                return None
            if len(successors) > 1:
                if np.random.random() <= 0.5:
                    mask_node = False
            if mask_node:
                masked_idx.append(child)
                new_nodes_were_added = True
                visited.add(child)
                queue.append(child)
                if len(masked_idx) == mask_size:
                    return masked_idx

def mask_subgraph_limit_size(adj, feat, max_size=25, use_max=True):
    if use_max: 
        mask_size = max_size
    else:
        mask_size = np.random.randint(5, max_size)
    while True:
        out = traverse_graph(adj, mask_size)
        if out is None:
#             print ('loop encountered?')
            continue
        else:
            masked_idx = out
            break

    ix_mesh = np.ix_(masked_idx, masked_idx)
    mask_adj = adj[ix_mesh].copy()
    mask_feat = feat[masked_idx].copy()
    Gmasked_adj = adj.copy()
    mask_outgoing = np.sum(Gmasked_adj[masked_idx], axis=0).reshape(1, -1)
    Gmasked_adj = np.append(Gmasked_adj, mask_outgoing, axis=0)
    
    mask_incoming = np.sum(Gmasked_adj[:, masked_idx], axis=1).reshape(-1, 1)
    Gmasked_adj = np.append(Gmasked_adj, mask_incoming, axis=1)
    Gmasked_adj = np.delete(Gmasked_adj, masked_idx, axis=0)
    Gmasked_adj = np.delete(Gmasked_adj, masked_idx, axis=1)
    
    masked_idx = np.asarray(masked_idx)
    Gmasked_feat = np.delete(feat.copy(), masked_idx, axis=0)
    Gmasked_feat = np.append(Gmasked_feat, np.zeros((Gmasked_feat.shape[0], 1)), axis=1)
    Gmasked_feat = np.append(Gmasked_feat, np.zeros((1, Gmasked_feat.shape[1])), axis=0)
    Gmasked_feat[-1, -1] = 1
    
    if not nx.is_directed_acyclic_graph(nx.from_numpy_array(mask_adj, create_using=nx.DiGraph())):
        print("the resulting graph is not DAG, try again")
        return mask_subgraph_limit_size(adj, feat, max_size)
    
    return mask_adj, mask_feat, Gmasked_adj, Gmasked_feat 

def mask_subgraph_limit_depth(adj, feat, max_depth=5, const=True):
    queue = [np.random.randint(adj.shape[0])]
    pending_queue = []
    masked_idx = [queue[0]]
    visited = set()
    
    if const:
        mask_size = max_depth
    else:
        mask_size = np.random.randint(1, max_depth)
    for j in range(mask_size):
        if len(masked_idx)>=20:
            break
        for i in queue:
            visited.add(i)
            successors = np.where(adj[i])[0]
            for n in successors:
                if n in masked_idx:
                    continue
                if n in visited:
                    # loop encountered, try again
                    print ('n: ', n, 'visited: ', visited)
                    print ('loop encountered?')
                    return mask_subgraph(adj, feat, max_depth)
                masked_idx.append(n)
                pending_queue.append(n)
        queue = pending_queue.copy()
        pending_queue = []
    ix_mesh = np.ix_(masked_idx, masked_idx)
    mask_adj = adj[ix_mesh].copy()
    mask_feat = feat[masked_idx].copy()
    Gmasked_adj = adj.copy()
    mask_outgoing = np.sum(Gmasked_adj[masked_idx], axis=0).reshape(1, -1)
    Gmasked_adj = np.append(Gmasked_adj, mask_outgoing, axis=0)
    
    mask_incoming = np.sum(Gmasked_adj[:, masked_idx], axis=1).reshape(-1, 1)
    Gmasked_adj = np.append(Gmasked_adj, mask_incoming, axis=1)
    Gmasked_adj = np.delete(Gmasked_adj, masked_idx, axis=0)
    Gmasked_adj = np.delete(Gmasked_adj, masked_idx, axis=1)
    
    masked_idx = np.asarray(masked_idx)
    Gmasked_feat = np.delete(feat.copy(), masked_idx, axis=0)
    Gmasked_feat = np.append(Gmasked_feat, np.zeros((Gmasked_feat.shape[0], 1)), axis=1)
    Gmasked_feat = np.append(Gmasked_feat, np.zeros((1, Gmasked_feat.shape[1])), axis=0)
    Gmasked_feat[-1, -1] = 1
    
    if not nx.is_directed_acyclic_graph(nx.from_numpy_array(mask_adj, create_using=nx.DiGraph())):
        print("the resulting graph is not DAG, try again")
        return mask_subgraph_limit_depth(adj, feat, max_depth)
    
    return mask_adj, mask_feat, Gmasked_adj, Gmasked_feat 

class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, feat_list, limit_depth=True, max_mask_size=25, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        self.max_mask_size = max_mask_size
        if limit_depth: 
            self.masking_func = mask_subgraph_limit_depth
        else:
            self.masking_func = mask_subgraph_limit_size
        self.feat_all = feat_list.copy()
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        G_adj = self.adj_all[idx].copy()
        G_feat = np.asarray(self.feat_all[idx].copy().todense())
        M_adj, M_feat, Gmasked_adj, Gmasked_feat = self.masking_func(G_adj, G_feat, self.max_mask_size)
        
        x = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x[0,:] = 0 # the first input token is all ones
        y = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        graph_size = M_adj.shape[0]
        x_idx = np.random.permutation(M_adj.shape[0])
        M_adj = M_adj[np.ix_(x_idx, x_idx)]
        M_adj_copy_matrix = np.asmatrix(M_adj)
        M = nx.from_numpy_matrix(M_adj_copy_matrix, create_using=nx.DiGraph())
        # then do topological sort in the permuted M
        x_idx = np.array(list(nx.algorithms.dag.topological_sort(M)))
#         print ("topologically sorted order", x_idx)
        M_adj = M_adj[np.ix_(x_idx, x_idx)]
        
        assert np.all(M_adj == np.triu(M_adj))
        M_adj = np.transpose(M_adj)
        M_adj_encoded = encode_adj(M_adj.copy(), max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y[0:M_adj_encoded.shape[0], :] = M_adj_encoded
        x[1:M_adj_encoded.shape[0] + 1, :] = M_adj_encoded

        return {
            # for GraphRNN
            'x':x,'y':y, 'len':M_adj.shape[0],
            # for regular GCN
            'G_adj':G_adj, 'G_feat':G_feat,
            # for masking GCN
            'Gmasked_adj':Gmasked_adj,
            'Gmasked_feat':Gmasked_feat
        }

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix,create_using=nx.DiGraph())
            # then do topological sort in the permuted G
            x_idx = np.array(topologicalSort(G))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy = np.transpose(adj_copy)
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node

    
class Graph_sampler_from_gcn_format(torch.utils.data.Dataset):
    def __init__(self):
        self.adj_all = None
        self.feat_all = None
            
    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        G_adj = self.adj_all[idx]
        G_feat = self.feat_all[idx]

        return {# for regular GCN
                'G_adj':G_adj, 'G_feat':G_feat}

    
    
class Graph_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, feat_list):
        self.adj_all = []
        self.feat_all = feat_list
        for G in G_list:
            self.adj_all.append(nx.to_scipy_sparse_matrix(G))
            
    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        G_adj = self.adj_all[idx]
        G_feat = self.feat_all[idx]

        return {# for regular GCN
                'G_adj':G_adj, 'G_feat':G_feat}


class Graph_sequence_sampler_flexible():
    def __init__(self, G_list):
        self.G_list = G_list
        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

        self.y_batch = []
    def sample(self):
        # generate input x, y pairs
        # first sample and get a permuted adj
        adj_idx = np.random.randint(len(self.adj_all))
        adj_copy = self.adj_all[adj_idx].copy()
        # print('graph size',adj_copy.shape[0])
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix,create_using=nx.DiGraph())
        # then do bfs in the permuted G
        x_idx = np.array(topologicalSort(G))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        # get the feature for the permuted G
        # dict = nx.bfs_successors(G, start_idx)
        # print('dict', dict, 'node num', self.G.number_of_nodes())
        # print('x idx', x_idx, 'len', len(x_idx))

        # print('adj')
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_copy.shape[0]):
        #     print(adj_copy[print_i].astype(int))
        # adj_before = adj_copy.copy()

        # encode adj
        adj_encoded = encode_adj_flexible(adj_copy.copy())
        # print('adj encoded')
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_copy.shape[0]):
        #     print(adj_copy[print_i].astype(int))


        # decode adj
        # print('adj recover error')
        # adj_decode = decode_adj(adj_encoded.copy(), max_prev_node=self.max_prev_node)
        # adj_err = adj_decode-adj_copy
        # print(np.sum(adj_err))
        # if np.sum(adj_err)!=0:
        #     print(adj_err)
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_err.shape[0]):
        #     print(adj_err[print_i].astype(int))

        # get x and y and adj
        # for small graph the rest are zero padded
        self.y_batch=adj_encoded


        # np.set_printoptions(linewidth=200,precision=3)
        # print('y\n')
        # for print_i in range(self.y_batch[i,:,:].shape[0]):
        #     print(self.y_batch[i,:,:][print_i].astype(int))
        # print('x\n')
        # for print_i in range(self.x_batch[i, :, :].shape[0]):
        #     print(self.x_batch[i, :, :][print_i].astype(int))
        # print('adj\n')
        # for print_i in range(self.adj_batch[i, :, :].shape[0]):
        #     print(self.adj_batch[i, :, :][print_i].astype(int))
        # print('adj_norm\n')
        # for print_i in range(self.adj_norm_batch[i, :, :].shape[0]):
        #     print(self.adj_norm_batch[i, :, :][print_i].astype(float))
        # print('feature\n')
        # for print_i in range(self.feature_batch[i, :, :].shape[0]):
        #     print(self.feature_batch[i, :, :][print_i].astype(float))

        return self.y_batch,adj_copy