import argparse
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle as pkl
import scipy.misc
import scipy.sparse as sp
import shutil
import time as tm
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import random

from tensorboard_logger import configure, log_value
from time import gmtime, strftime
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
from random import shuffle

from encoder import GraphConvolution
from utils import construct_gcn_batch_masked, csr_to_torch_sparse

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nhid3)
        self.gc4 = GraphConvolution(nhid3, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = self.gc4(x, adj)
        return x
    
class FC(nn.Module):
    def __init__(self, nfeat, nhid1, nout):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid1)
        self.fc2 = nn.Linear(nhid1, nout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def run_epoch(fc, gcn, optimizer_fc, optimizer_gcn, data, 
              writer, loss_log_str, log_i,
              total_acc_log_str, masked_acc_log_str, 
              print_every=50, train=True):
    epoch_loss = 0
    epoch_acc_total = 0
    epoch_acc_masked = 0
    if train:
        fc.train()
        gcn.train()
    else:
        fc.eval()
        gcn.eval()
    adj_all, feat_all, masked_feat_all, masks = data
    for batch_idx, batch in enumerate(construct_gcn_batch_masked(adj_all, 
                                                                 feat_all, 
                                                                 masked_feat_all,
                                                                 masks,
                                                                 batch_size=1)):
        if train:
            gcn.zero_grad()
            fc.zero_grad()
        adj, feat, masked_feat, mask = batch
        if adj.shape[0] > 15000:
            continue
        adj_v = Variable(torch.FloatTensor(adj.todense())).cuda()
        feat_v = Variable(torch.FloatTensor(masked_feat.todense())).cuda()
        mask_v = Variable(torch.LongTensor(mask).cuda())
        n_masked = sum(mask)
        
        labels = np.asarray(np.argmax(feat, 1)).reshape(-1, 1).flatten()
        labels_v = Variable(torch.LongTensor(labels)).cuda()
        
        gcn_out = gcn(feat_v, adj_v)

        y_pred = fc(gcn_out)
        loss = F.cross_entropy(y_pred, labels_v, reduction='none')
        loss = (loss * mask_v).sum()/n_masked
        
        if train:
            loss.backward()
            optimizer_fc.step()
            optimizer_gcn.step()
        
        epoch_loss += loss.data.item()
        acc = (torch.argmax(y_pred, 1).eq(labels_v))
        acc_masked = (acc * mask_v).sum().data.item()/n_masked
        acc_total = acc.sum().data.item()/(labels_v.shape[0])
        epoch_acc_total += acc_total
        epoch_acc_masked += acc_masked
        
        if train:
            writer.add_scalar("Batch " + loss_log_str, loss, log_i)
            writer.add_scalar("Memory", torch.cuda.memory_allocated(0), log_i)
            writer.add_scalar("Batch " + total_acc_log_str, acc_total, log_i)
            writer.add_scalar("Batch " + masked_acc_log_str, acc_masked, log_i)
            log_i += 1
        
    epoch_loss /= len(adj_all)
    epoch_acc_total /= len(adj_all)
    epoch_acc_masked /= len(adj_all)
    return epoch_loss, epoch_acc_total, epoch_acc_masked, log_i
    
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cuda', dest='cuda', type=int,
                        help='cuda device number')
    parser.add_argument('--mask_p', dest='mask_p', type=float,
                        help='percentage of masked nodes')

    args = parser.parse_args()
    with open ('/nas/home/shushan/updated_graphs/fold_0/old_gcn_on_oj_test.pkl', 'rb') as f:
        test = pkl.load(f)
    with open ('/nas/home/shushan/updated_graphs/fold_0/old_gcn_on_oj_train.pkl', 'rb') as f:
        train = pkl.load(f)
    with open ('/nas/home/shushan/updated_graphs/fold_0/old_gcn_on_oj_val.pkl', 'rb') as f:
        val = pkl.load(f)

    train_adj, train_feat, train_labels = train
    test_adj, test_feat, test_labels = test
    val_adj, val_feat, val_labels = val


    with open('/nas/home/shushan/updated_graphs/fold_0/node_label_idx_dict_old.pkl', 'rb') as f:
        dicts = pkl.load(f, encoding='latin1')
    node_label_idx_dict, node_label_freq = dicts

    torch.cuda.set_device(args.cuda)
    
    masked_dir = '/nas/home/shushan/updated_graphs/masked_{}_fold_0/'.format(args.mask_p)
    with open(masked_dir + 'old_gcn_on_oj_train.pkl', 'rb') as f:
        masked_train, masks_train = pkl.load(f)
        masked_train = np.asarray(masked_train)
    with open(masked_dir + 'old_gcn_on_oj_val.pkl', 'rb') as f:
        masked_val, masks_val = pkl.load(f)
        masked_val = np.asarray(masked_val)

    E = 32
    gcn = GCN(nfeat=masked_train[0].shape[1], nhid1=64,
              nhid2=64, nhid3=64, nclass=E).cuda()
    optimizer_gcn = optim.Adam(list(gcn.parameters()), lr=0.001)
    fc = FC(nfeat=E, nhid1=32, nout=train_feat[0].shape[1]).cuda()
    optimizer_fc = optim.Adam(list(fc.parameters()), lr=0.001)

    writer = SummaryWriter()
    task_str = 'GCN reconstruction mask {}p old repr '.format(args.mask_p)
    loss_log_str = task_str + 'loss/'
    total_acc_log_str = task_str + 'acc/total '
    masked_acc_log_str = task_str + 'acc/masked '
    
    max_epochs = 5
    log_i = 0
    for epoch in range(max_epochs):
        out = run_epoch(data=(train_adj, train_feat, masked_train, masks_train), 
                        gcn=gcn, optimizer_gcn=optimizer_gcn,
                        fc=fc,  optimizer_fc=optimizer_fc, 
                        writer=writer, log_i=log_i,
                        loss_log_str = loss_log_str + 'train',
                        total_acc_log_str = total_acc_log_str + 'train', 
                        masked_acc_log_str = masked_acc_log_str + 'train', 
                        train=True)
        epoch_loss, epoch_acc_total, epoch_acc_masked, log_i = out
        writer.add_scalar(loss_log_str + 'train', epoch_loss, epoch)
        writer.add_scalar(total_acc_log_str + 'train', epoch_acc_total, epoch)
        writer.add_scalar(masked_acc_log_str + 'train', epoch_acc_masked, epoch)

        out = run_epoch(data=(val_adj, val_feat, masked_val, masks_val), 
                        gcn=gcn, optimizer_gcn=optimizer_gcn,
                        fc=fc,  optimizer_fc=optimizer_fc, 
                        writer=writer, log_i=-1,
                        loss_log_str = None,
                        total_acc_log_str = None,
                        masked_acc_log_str = None,
                        train=False)
        epoch_loss, epoch_acc_total, epoch_acc_masked, _ = out
        writer.add_scalar(loss_log_str + 'val', epoch_loss, epoch)
        writer.add_scalar(total_acc_log_str + 'val', epoch_acc_total, epoch)
        writer.add_scalar(masked_acc_log_str + 'val', epoch_acc_masked, epoch)

if __name__ == '__main__':
    main()

