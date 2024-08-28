import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import progressbar
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from collections import defaultdict
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from utils import *
from rnn import *
from tree import *
sys.stdout = open('rnn_training_out_pt2.txt','wt')

import pickle as pkl
with open('trees.pkl', 'rb') as f:
    trees = pkl.load(f)
    
def run_epoch(model, optimizer, train_data, print_every=1000000):
    loss_history = []
    train_acc_history = []
    
    random.shuffle(train_data)
    
#     pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(train_data)).start()
    for step, tree in enumerate(train_data):
        _, loss = model.getLoss(tree)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 5, norm_type=2.)
        optimizer.step()
        loss_history.append(loss.data.item())

        if step > 0 and step % print_every == 0:
            print ('\r{}/{} : loss = {}'.format(
                    step, len(train_data), np.mean(loss_history)))
            # to save time, evaluate on previous 300 examples in train data only
            if step > 300:
                begin = step - 300
            else:
                begin = 0
            train_acc = model.evaluate(train_data[begin : step])
            train_acc_history.append(train_acc)
            print ('\r{}/{} : train acc = {}'.format(
                    step, len(train_data), train_acc))
            sys.stdout.flush()        
#         pbar.update(step)
#     pbar.finish()    
    return loss_history, train_acc_history

def train(model, train_data, val_data):
#     widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    
    bestAcc = 0
    
    complete_loss_history = []
    complete_train_acc_history = []
    val_acc_history = []

    print_every = 5000
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.lr, 
                                 weight_decay=model.config.l2, amsgrad=model.config.amsgrad)
    for epoch in range(model.config.max_epochs):
        print("Epoch %d" % epoch)
        loss_history, train_acc_history = run_epoch(model, optimizer, train_data, print_every)
        complete_loss_history.extend(loss_history)
        complete_train_acc_history.extend(train_acc_history)
        
        acc = model.evaluate(val_data)
        if bestAcc < acc: 
            bestAcc = acc
            torch.save(model.state_dict(), 'weights/rnn_embedsize={}_l2={}_lr={}.ckp_{}'.format(
                model.config.embed_size, model.config.l2, model.config.lr, epoch))

        print("Validation Acc:" + str(round(acc, 2)) + \
              "(best:" + str(round(bestAcc, 2)) + ")")
        sys.stdout.flush()
    plt.plot(complete_loss_history)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_history.png")
    #plt.show()
    
    
def grid_search_models():
    embed_sizes = [25, 50, 100, 350]
    l2 = [0, 1e-2, 1e-4]
    lr = [1e-4, 1e-3, 1e-2, 1e-1]
    amsgrad = [False, True]

    hyperparameters_tried = [
        {'a':False, 'l2':0, 'lr':1e-4, 'e':25},
        {'a':False, 'l2':0, 'lr':1e-4, 'e':50},
        {'a':False, 'l2':0, 'lr':1e-4, 'e':100},
    ]
    
    train_data, val_data, test_data = splitData(trees)
    #train_data = train_data[:10]
    #val_data = val_data[:10]
    
    vocab = Vocab()
    train_words = [t.getWords() for t in train_data]
    vocab.construct(list(itertools.chain.from_iterable(train_words)))
    print ('Starting training') 
    for a_i in amsgrad:
        for l2_i in l2:
            for lr_i in lr:
                for e_i in embed_sizes:
                    h = {'a':a_i, 'l2':l2_i, 'lr':lr_i, 'e':e_i}
                    if h in hyperparameters_tried:
                        continue
                    config = Config()
                    config.embed_size = e_i
                    config.lr = lr_i
                    config.l2 = l2_i
                    config.amsgrad = a_i
                    print ('~~~~~~~~~~~~~~~~~~~~~~')
                    print ('lr = {}, l2 = {}, amsgrad = {}, embed_size = {}'.format(
                    lr_i, l2_i, a_i, e_i))
                    print ('~~~~~~~~~~~~~~~~~~~~~~')

                    if CUDA: model = RecursiveNN(vocab, config).cuda()
                    else: model = RecursiveNN(vocab, config)
                    train(model, train_data, val_data)
grid_search_models()
