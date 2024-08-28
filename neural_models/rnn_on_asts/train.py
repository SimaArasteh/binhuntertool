import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
import random

from tree import Tree
from utils import splitData
from models import Config, Var

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# TODO: figure out how to install tensorboardX

# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
# def plot_and_print(step, print_every, loss_history, train_acc_history, N):
    
#     print ('\r{}/{} : loss = {}'.format(step, N, np.mean(loss_history)))
#     writer.add_scalar('data/train_loss', loss_history[-1], step)
    
#     if step % print_every == 0:
#         begin = max(0, step - 100)
#         train_acc = model_evaluate(model, train_data[begin : step])
#         train_acc_history.append(train_acc)
        
#         print ('\r{}/{} : train acc = {}'.format(step, N, train_acc))
        
#         writer.add_scalar('data/train_acc', train_acc, step)
    
#     return train_acc


class SuperconvergenceLR(object):
    def __init__(self, optimizer, lowest_lr, highest_lr, max_epochs):
        self.optimizer = optimizer
        print ('Using superconvergence lr scheduler, \
               highest lr: {}, lowest lr: {}'.format(highest_lr, lowest_lr))
        self.inc_period = max_epochs//2
        self.delta_lr = (highest_lr - lowest_lr)/self.inc_period
        self.last_epoch = -1
        self.lr = lowest_lr
        
    def get_lr(self):
        if self.last_epoch >= self.inc_period:
            self.lr -= self.delta_lr
            print('new lr: ', self.lr)
            return [self.lr]
        self.lr += self.delta_lr
        print('new lr: ', self.lr)
        return [self.lr]

    def step(self):
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def model_evaluate(model, trees):
    correct = 0.0
    n = len(trees)
    losses = []
    for j, tree in enumerate(trees):
        prediction, loss = model.forward(tree)
        losses.append(loss.data.item())
        correct += (prediction.data == Var(torch.tensor([tree.label-1]))).sum()
    return correct.data.item() / n, losses


def run_epoch(model, optimizer, train_data, print_every=1000, print_acc=False):
    loss_history = []
    train_acc_history = []
    
    random.shuffle(train_data)
    
    start_time = time.time()
    for step, tree in enumerate(train_data):
        _, loss = model.forward(tree)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.)
       
        optimizer.step()
        loss_history.append(loss.data.item())
        if step > 0 and step % print_every == 0:
            end_time = time.time()
            print ('\r{}/{} : loss = {}, time = {}'.format(
                    step, len(train_data), np.mean(loss_history[step - print_every : step]), 
                    end_time - start_time))
            if print_acc:
                train_acc = model_evaluate(model, train_data[step - print_every : step])
                train_acc_history.append(train_acc)
                print ('\r{}/{} : train acc = {}'.format(
                            step, len(train_data), train_acc))
            sys.stdout.flush()
            start_time = time.time()
         
    print('Epoch completed, avg loss: ', np.mean(loss_history))
    return loss_history, train_acc_history

def train(model, model_prefix, train_data, val_data, sc, print_every=5000):
    bestAcc = 0
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    model_name = model_prefix + 'rnn_embed_size={}_l2={}_lr={}'.format(
                           model.config.embed_size, 
                           model.config.l2, 
                           model.config.lr)


    now = datetime.datetime.now()
    weights_folder = 'weights/' + str(now.strftime('%Y-%m-%d'))
    
    if not os.path.isdir(weights_folder):
        os.mkdir(weights_folder)
    
    optimizer = torch.optim.RMSprop(model.parameters(), 
                                 lr=model.config.lr)
    if sc:
#         model.config.highest_lr = model.config.lr * 100
#         lr_scheduler = SuperconvergenceLR(optimizer=optimizer, 
#                                           lowest_lr=model.config.lr, 
#                                           highest_lr=model.config.highest_lr,
#                                           max_epochs=model.config.max_epochs)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                                         patience=0,
                                         factor=0.6, 
                                         verbose=True)
#         lr_scheduler = StepLR(optimizer=optimizer, 
#                               step_size=1,
#                               gamma=1.1)
    
    for epoch in range(model.config.max_epochs):
        print("Epoch %d" % epoch)
        start_time = time.time()
        _, _ = run_epoch(model, optimizer, train_data, print_every)
        end_time = time.time()
        
        print("Epoch completed in {} mins".format((end_time - start_time)//60))
        sys.stdout.flush()
        
        idx = np.random.choice(range(len(train_data)), 1000)
        acc, loss = model_evaluate(model, train_data[idx])
        loss = sum(loss)/len(loss)
        print("Train Acc: {}, loss: {}".format(
            str(round(acc, 2)), 
            str(round(loss, 2))))
        train_loss_history.append(loss)

        acc, loss = model_evaluate(model, val_data)
        loss = sum(loss)/len(loss)
        print("Validation Acc: {} (best: {}), loss: {}".format(
            str(round(acc, 2)),
            str(round(bestAcc, 2)), 
            str(round(loss, 2))))
        val_loss_history.append(loss)
        sys.stdout.flush()
        
        if sc:
            lr_scheduler.step(loss)
        
        if bestAcc < acc: 
            bestAcc = acc
            torch.save(model.state_dict(), 
                       weights_folder + '/{}.ckp_{}'.format(model_name, epoch))
        
    
    plt.plot(train_loss_history, '-o')
    plt.plot(val_loss_history, '-o')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('{}.loss_history.png'.format(model_name))
    
def grid_search_models(params, trees, 
                       model_prefix, sc, trimmed, 
                       model_name, model_weights=None):
    embed_sizes = params['embed_sizes']
    l2 = params['l2']
    lr = params['lr']
    amsgrad = params['amsgrad']
    
    train_data, val_data, test_data = splitData(trees)
    
    if trimmed:
        from trimmed_vocab import TrimmedVocab
        vocab = TrimmedVocab()
    else:
        from vocab import Vocab
        vocab = Vocab()

    vocab.construct_from_trees(train_data)
    
    for a_i in amsgrad:
        for l2_i in l2:
            for lr_i in lr:
                for e_i in embed_sizes:
                    config = Config()
                    config.embed_size = e_i
                    config.lr = lr_i
                    config.l2 = l2_i
                    config.amsgrad = a_i
                    config.max_epochs = 50
                    
                    print ('~~~~~~~~~~~~~~~~~~~~~~')
                    print ('lr = {}, l2 = {}, amsgrad = {}, embed_size = {}'.format(
                    lr_i, l2_i, a_i, e_i))
                    print ('~~~~~~~~~~~~~~~~~~~~~~')
            
                    if model_name == 'RecursiveNN':
                        from models import RecursiveNN
                        model = RecursiveNN(vocab, config).cuda()
                    elif model_name == 'RecursiveNN_BN':
                        from models import RecursiveNN_BN
                        model = RecursiveNN_BN(vocab, config).cuda()
                    elif model_name == 'MultiplicativeRecursiveNN':
                        from models import MultiplicativeRecursiveNN
                        model = MultiplicativeRecursiveNN(vocab, config).cuda()
                    elif model_name == 'ResidualRecursiveNN':
                        from models import ResidualRecursiveNN
                        model = ResidualRecursiveNN(vocab, config).cuda()
                    elif model_name == 'ResidualRecursiveNN_w_N':
                        from models import ResidualRecursiveNN_w_N
                        model = ResidualRecursiveNN_w_N(vocab, config).cuda()
                    else:
                        raise "No valid model specified: {}".format(model_name) 
                    if model_weights is not None: 
                        print ('Loading pre-trained parameters from {}'.format(model_weights))
                        model.load_state_dict(torch.load(model_weights))
                    train(model, model_prefix, train_data, val_data, sc)
