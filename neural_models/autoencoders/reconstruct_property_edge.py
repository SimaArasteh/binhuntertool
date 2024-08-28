import torch 
torch.cuda.set_device(0)

import sys
import logging
import networkx as nx
import numpy as np
import pickle as pkl
import time as tm
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append('/nas/home/shushan/binary-reconstruction/analysis/neural_models/autoencoders/graph_generation')
sys.path.append('/nas/home/shushan/binary-reconstruction/analysis/neural_models/autoencoders/pygcn')
from graph_generation.args import Args
from models import GCN, FC 
from train import train_reconstruct_property_edge as train_epoch
from train import test_reconstruct_property_edge as test_epoch

args = Args()
args.epochs = 30000
time_all = np.zeros(args.epochs);
epoch = 0
args.epochs_log = 1

with open ('/nas/home/shushan/gcn_on_oj_cv_0_test.pkl', 'rb') as f:
    test = pkl.load(f)
with open ('/nas/home/shushan/gcn_on_oj_cv_0_train.pkl', 'rb') as f:
    train = pkl.load(f)
with open ('/nas/home/shushan/gcn_on_oj_cv_0_val.pkl', 'rb') as f:
    val = pkl.load(f)

train_adj, train_feat, _ = train
test_adj, test_feat, _ = test
val_adj, val_feat, _ = val

gcn = GCN(nfeat=train_feat[0].shape[1], nhid1=1000,
          nhid2=500, nhid3=200, nclass=100, 
          dropout=False, name='gcn').cuda()
fc = FC(nfeat=200, nhid1=50, nout=2).cuda()

optimizer_gcn = optim.Adam(list(gcn.parameters()), lr=5e-4)
scheduler_gcn = MultiStepLR(optimizer_gcn, milestones=args.milestones, gamma=args.lr_rate)
optimizer_fc = optim.Adam(list(fc.parameters()), lr=5e-4)
scheduler_fc = MultiStepLR(optimizer_fc, milestones=args.milestones, gamma=args.lr_rate)

rolling_loss = []
while epoch<=args.epochs:
    l, t = train_epoch(epoch, args, gcn, fc, (train_adj, train_feat), 
                       optimizer_gcn, optimizer_fc, 
                       scheduler_gcn, scheduler_fc)
    rolling_loss.append(l)
    if epoch % args.epochs_log == 0:
        with open('reconstruct_property_edge_training_losses.txt', 'w') as f:
            f.write(str(rolling_loss))
        print('Epoch: {}/{}, train loss: {:.4f} in {} secs'.format(
            epoch, args.epochs, np.mean(rolling_loss[-100:]), t))
        l = test_epoch(epoch, args, gcn, fc, (val_adj, val_feat))
        print('Epoch: {}/{}, validation loss: {:.4f}'.format(
            epoch, args.epochs, l))
        with open('reconstruct_property_edge_test_losses.txt', 'a+') as f:
            f.write(str(l) + '\n')
    epoch += 1
