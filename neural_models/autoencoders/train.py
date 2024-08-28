import logging
import networkx as nx
import numpy as np
import pickle
import scipy.misc
import scipy.sparse as sp
import time as tm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
from tensorboard_logger import configure, log_value
from time import gmtime, strftime
from random import shuffle

from decoder import *
from utils import construct_gcn_batch, csr_to_torch_sparse
    
def train_epoch(epoch, args, encoder, decoder, data_loader,
                optimizer_enc, optimizer_dec, 
                scheduler_enc, scheduler_dec):
    start_time = time.time()
    gcn, masked_gcn = encoder
    rnn, output = decoder
    
    optimizer_gcn, optimizer_masked_gcn = optimizer_enc
    scheduler_gcn, scheduler_masked_gcn = scheduler_enc
    optimizer_rnn, optimizer_output = optimizer_dec
    scheduler_rnn, scheduler_output = scheduler_dec
    
    
    gcn.train()
    masked_gcn.train()
    rnn.train()
    output.train()
    
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        gcn.zero_grad()
        masked_gcn.zero_grad()
        rnn.zero_grad()
        output.zero_grad()
        
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))
        
        
        #TODO: add batch construction!
        batch_adj = Variable(torch.FloatTensor(data['G_adj'][0].float())).cuda()
        batch_feat = Variable(torch.FloatTensor(data['G_feat'][0].float())).cuda()
        gcn_out = gcn(batch_feat,batch_adj)
        
        batch_masked_adj = Variable(torch.FloatTensor(data['Gmasked_adj'][0].float())).cuda()
        batch_masked_feat = Variable(torch.FloatTensor(data['Gmasked_feat'][0].float())).cuda()
        masked_gcn_out = masked_gcn(batch_masked_feat, batch_masked_adj) 

        z_unsorted = torch.sum(gcn_out, dim=0)
        masked_z_unsorted = torch.sum(masked_gcn_out, dim=0)

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        
        # TODO: this will be necessary with batch_size > 1 and pooling
#         siV = Variable(sort_index.data).cuda()
#         z = torch.index_select(z_unsorted,0,siV)
#         masked_z = torch.index_select(masked_z_unsorted,0,siV)

        combined_z = torch.cat((z_unsorted, masked_z_unsorted), 0)
        combined_z = combined_z.unsqueeze(0)
        combined_z = combined_z.unsqueeze(0)
        
        x = Variable(torch.FloatTensor(x), requires_grad=True).cuda()
        zeros = torch.zeros((x.shape[0], x.shape[1] - 1, x.shape[2]), requires_grad=True).cuda() 
        z_ext = torch.cat((combined_z, zeros), dim=1) # [1, 1372, 100]
        x = x + z_ext

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = F.binary_cross_entropy(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_gcn.step()
        optimizer_masked_gcn.step()
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_gcn.step()
        scheduler_masked_gcn.step()
        scheduler_output.step()
        scheduler_rnn.step()  


#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss.data.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
#         log_value('loss_'+args.fname, loss.data.item(), epoch*args.batch_ratio+batch_idx)
        feature_dim = y.size(1)*y.size(2)
#         print (feature_dim)
        loss_sum += loss.data.item()*feature_dim
    return loss_sum/(batch_idx+1)


def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def train_reconstruct_property_edge_direction(epoch, args, gcn, fc, 
                                              data, optimizer_gcn,
                                              optimizer_fc, scheduler_gcn, 
                                              scheduler_fc, 
                                              num_examples=8):
    start_time = time.time()
    
    gcn.train()
    fc.train()
    loss_sum = 0
    batch_size = 32

    positive = Variable(torch.FloatTensor([[0, 1]])).cuda()
    negative = Variable(torch.FloatTensor([[1, 0]])).cuda()
    adj_all, features_all = data
    
    for batch_idx, batch in enumerate(construct_gcn_batch(adj_all, features_all, 
                                                          batch_size=batch_size)):
        adj, feat = batch
        
        adj_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(adj))).cuda()
        feat_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(feat))).cuda()
        gcn_out = gcn(feat_v, adj_v)
        
        idx = torch.LongTensor(adj.nonzero()).cuda()
        loss_ = 0
        if num_examples < 1:
            if num_examples <= 0:
                select_idx = np.arange(idx.shape[1])
            else:
                select_idx = np.random.choice(np.arange(idx.shape[1]), int(num_examples * idx.shape[1]))
        else:
            select_idx = np.random.choice(np.arange(idx.shape[1]), num_examples*batch_size)
        for i in select_idx:
            gcn.zero_grad()
            fc.zero_grad()
            if np.random.random() > 0.5:
                a = torch.cat((torch.index_select(gcn_out, 0, idx[0, i]), 
                               torch.index_select(gcn_out, 0, idx[1, i])), 1)
                y_pred = fc(a)
#                 print ('y_pred for correct direction: ', y_pred)
                loss = F.binary_cross_entropy(y_pred, positive)
            else:
                a = torch.cat((torch.index_select(gcn_out, 0, idx[1, i]), 
                               torch.index_select(gcn_out, 0, idx[0, i])), 1)
                
                y_pred = fc(a)
#                 print ('y_pred for flipped direction: ', y_pred)
                loss = F.binary_cross_entropy(y_pred, negative)
            loss_ += loss.data.item()
            loss.backward(retain_graph=True)
            
            optimizer_gcn.step()
            optimizer_fc.step()
            scheduler_gcn.step()
            scheduler_fc.step()
        
        loss_ /= (i + 1)
        if batch_idx % 100 == 0:
            print ("Completed {} batches, train loss: {:.4f}".format(batch_idx + 1, loss_))
            print ("Time elapsed {}s".format(int(time.time() - start_time)))
        loss_sum += loss_


    return loss_sum/(batch_idx+1), time.time() - start_time


def test_reconstruct_property_edge_direction(epoch, args, gcn, fc, data, num_examples=16):
    
    start_time = time.time()
    
    gcn.eval()
    fc.eval()
    
    loss_sum = 0
    adj_all, features_all = data
    for batch_idx, batch in enumerate(construct_gcn_batch(adj_all, features_all)):
        adj, feat = batch

        adj_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(adj))).cuda()
        feat_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(feat))).cuda()        
        gcn_out = gcn(feat_v, adj_v)

        idx = torch.LongTensor(adj.nonzero()).cuda()
        loss_ = 0
        for i in range(idx.shape[1]):
            if np.random.random() > 0.5:
                a = torch.cat((torch.index_select(gcn_out, 0, idx[0, i]), 
                               torch.index_select(gcn_out, 0, idx[1, i])), 1)
                y = Variable(torch.FloatTensor([[0, 1]])).cuda()
            else:
                a = torch.cat((torch.index_select(gcn_out, 0, idx[1, i]), 
                               torch.index_select(gcn_out, 0, idx[0, i])), 1)
                y = Variable(torch.FloatTensor([[1, 0]])).cuda()
            y_pred = fc(a)
            loss = F.binary_cross_entropy(y_pred, y)
            loss_ += loss.data.item()
            
        loss_ /= (i + 1)
        loss_sum += loss_
    return loss_sum/(batch_idx+1)
    
def train_reconstruct_property_edge(epoch, args, gcn, fc, data, optimizer_gcn,
                optimizer_fc, scheduler_gcn, scheduler_fc, num_examples=100):
    start_time = time.time()
    
    gcn.train()
    fc.train()
    loss_sum = 0
    adj_all, feat_all = data
    
    positive = torch.FloatTensor([[0, 1]]).cuda()
    negative = torch.FloatTensor([[1, 0]]).cuda()

    for batch_idx, data in enumerate(construct_gcn_batch(adj_all, feat_all)):
        gcn.zero_grad()
        fc.zero_grad()
        
        adj, feat = data

        adj_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(adj))).cuda()
        feat_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(feat))).cuda()        
        gcn_out = gcn(feat_v, adj_v)

        idx = torch.LongTensor(adj.nonzero()).cuda()
        loss = None

        edges_idx = np.random.choice(np.arange(len(idx[0])), num_examples)
        for i in edges_idx:
            edge = torch.cat((torch.index_select(gcn_out, 0, idx[0, i]), 
                              torch.index_select(gcn_out, 0, idx[1, i])), 1)
            y_pred = fc(edge)
            if loss is None:
                loss = F.binary_cross_entropy(y_pred, positive)
            else:
                loss += F.binary_cross_entropy(y_pred, positive)
        non_edges_idx = 0
        while non_edges_idx < num_examples:
            i = np.random.choice(adj.shape[0], 2)
            if adj[i[0], i[1]] == 0:
                i = torch.LongTensor(i).cuda()
                non_edges_idx += 1
                non_edge = torch.cat((torch.index_select(gcn_out, 0, i[0]), 
                                      torch.index_select(gcn_out, 0, i[1])), 1)
                y_pred = fc(non_edge)
                if loss is None:
                    loss = F.binary_cross_entropy(y_pred, negative)
                else:
                    loss += F.binary_cross_entropy(y_pred, negative)
        loss_sum += loss.data.item() / (num_examples * 2)
        loss.backward()

        optimizer_gcn.step()
        optimizer_fc.step()
        scheduler_gcn.step()
        scheduler_fc.step()

    return loss_sum/(batch_idx+1), time.time() - start_time

def test_reconstruct_property_edge(epoch, args, gcn, fc, data, num_examples=1000):
    start_time = time.time()
    adj_all, feat_all = data
    gcn.eval()
    fc.eval()
    loss_sum = 0
    positive = Variable(torch.FloatTensor([[0, 1]])).cuda()
    negative = Variable(torch.FloatTensor([[1, 0]])).cuda()

    for batch_idx, data in enumerate(construct_gcn_batch(adj_all, feat_all)):
        adj, feat = data

        adj_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(adj))).cuda()
        feat_v = Variable(torch.sparse.FloatTensor(*csr_to_torch_sparse(feat))).cuda()        
        gcn_out = gcn(feat_v, adj_v)

        idx = torch.LongTensor(adj.nonzero()).cuda()
        loss = 0

        edges_idx = np.random.choice(np.arange(len(idx[0])), 10)
        for i in edges_idx:
            edge = torch.cat((torch.index_select(gcn_out, 0, idx[0, i]), 
                              torch.index_select(gcn_out, 0, idx[1, i])), 1)
            y_pred = fc(edge)
            loss += F.binary_cross_entropy(y_pred, positive).data.item()
        non_edges_idx = 0
        while non_edges_idx < num_examples:
            i = np.random.choice(adj.shape[0], 2)
            if adj[i[0], i[1]] == 0:
                i = torch.LongTensor(i).cuda()
                non_edges_idx += 1
                non_edge = torch.cat((torch.index_select(gcn_out, 0, i[0]), 
                                      torch.index_select(gcn_out, 0, i[1])), 1)
                y_pred = fc(non_edge)
                loss += F.binary_cross_entropy(y_pred, negative).data.item()
        loss_sum += loss / (num_examples * 2)
        print ("Testing batch loss: ", loss / (num_examples * 2))
    return loss_sum/(batch_idx+1)
    
