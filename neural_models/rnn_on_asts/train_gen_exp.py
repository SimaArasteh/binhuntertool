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
    return round(correct.data.item() / n, 2), losses


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

        
def report_result(model, data, split, p=1000):
    if p > 0:
        idx = np.random.choice(range(len(data)), p)
        data = data[idx]
    
    acc, loss = model_evaluate(model, data)
    loss = round(sum(loss)/len(loss), 2)
    print("{} Acc: {}, loss: {}".format(split, acc, loss))
    
    sys.stdout.flush()
    return acc, loss


def fine_tune(gen_model, gtrain, val, print_every):
    train_loss_history = []
    val_loss_history = []
    
    optimizer = torch.optim.RMSprop(list(filter(lambda p: p.requires_grad, gen_model.parameters())), 
                                    lr=gen_model.config.lr)
    
    for ft_epoch in range(50):
#         print ("Fine-tuning epoch {}".format(ft_epoch))
#         start_time = time.time()
        _, _ = run_epoch(gen_model, optimizer, gtrain, print_every)
#         end_time = time.time()
#         print("Fine-tuning epoch completed in {} mins".format((end_time - start_time)//60))
#         sys.stdout.flush()
        
    train_loss_history.append(report_result(gen_model, gtrain, "Fine tuning train", p=-1))
    val_loss_history.append(report_result(gen_model, val, "Fine tuning val", p=-1))
    return train_loss_history, val_loss_history
        

def train(model, gen_model, model_prefix, ctrain, ctest, gtrain, gval, gtest, sc=False, print_every=5000):
    bestAcc = 0
    
    ctrain_loss_history = []
    gtrain_loss_history = []
    cval_loss_history = []
    gval_loss_history = []
    
    model_name = model_prefix + '_embed_size={}_l2={}_lr={}'.format(
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
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                                         patience=0,
                                         factor=0.6, 
                                         verbose=True)
    
    for epoch in range(model.config.max_epochs):
        print("Epoch %d" % epoch)
        start_time = time.time()
        _, _ = run_epoch(model, optimizer, ctrain, print_every)
        end_time = time.time()
        print("Epoch completed in {} mins".format((end_time - start_time)//60))
        sys.stdout.flush()
        
        acc, loss = report_result(model, ctrain, "Train", p=1000)
        ctrain_loss_history.append(loss)
        acc, loss = report_result(model, ctest, "Val", p=-1)
        cval_loss_history.append(loss)

        if epoch != 0 and epoch % 4 == 0:
            gen_model.set_weights_for_gen_exp(model)
            gtrain_loss_history_i, gval_loss_history_i = fine_tune(gen_model, 
                                                                   gtrain,
                                                                   gval, 
                                                                   print_every)
        gtrain_loss_history.extend(gtrain_loss_history_i)
        gval_loss_history.extend(gval_loss_history_i)
        
        if sc:
            lr_scheduler.step(loss)
        
        if bestAcc < acc: 
            bestAcc = acc
            torch.save(model.state_dict(), 
                       weights_folder + '/{}.ckp_{}'.format(model_name, epoch))
        
    
    plt.plot(ctrain_loss_history, '-o')
    plt.plot(cval_loss_history, '-o')
    plt.plot(gtrain_loss_history, '-x')
    plt.plot(gval_loss_history, '-x')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('{}.loss_history.png'.format(model_name))
    
    
def grid_search_models_gen_exp(params, trees, 
                               model_prefix, sc, trimmed, 
                               model_name, model_weights=None):
    embed_sizes = params['embed_sizes']
    l2 = params['l2']
    lr = params['lr']
    amsgrad = params['amsgrad']
    
    ctrain, ctest, gtrain, gval, gtest = generalizationExperimentSplitData(trees)
    clabel_size = len(np.unique(getLabels(ctrain)))
    glabel_size = len(np.unique(getLabels(gtrain)))
    
    if trimmed:
        from trimmed_vocab import TrimmedVocab
        vocab = TrimmedVocab()
    else:
        from vocab import Vocab
        vocab = Vocab()

    vocab.construct_from_trees(ctrain)
    
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
                    config.label_size = clabel_size
                    
                    gen_config = copy.deepcopy(config)
                    gen_config.label_size = clabel_size + glabel_size
                    
                    print ('~~~~~~~~~~~~~~~~~~~~~~')
                    print ('lr = {}, l2 = {}, amsgrad = {}, embed_size = {}'.format(
                    lr_i, l2_i, a_i, e_i))
                    print ('~~~~~~~~~~~~~~~~~~~~~~')
            
                    if model_name == 'RecursiveNN':
                        from models import RecursiveNN
                        model = RecursiveNN(vocab, config).cuda()
                        gen_model = RecursiveNN(vocab, gen_config)
                    elif model_name == 'RecursiveNN_BN':
                        from models import RecursiveNN_BN
                        model = RecursiveNN_BN(vocab, config).cuda()
                        gen_model = RecursiveNN_BN(vocab, gen_config)
                    elif model_name == 'MultiplicativeRecursiveNN':
                        from models import MultiplicativeRecursiveNN
                        model = MultiplicativeRecursiveNN(vocab, config).cuda()
                        gen_model = MultiplicativeRecursiveNN(vocab, gen_config)
                    elif model_name == 'ResidualRecursiveNN':
                        from models import ResidualRecursiveNN
                        model = ResidualRecursiveNN(vocab, config).cuda()
                        gen_model = ResidualRecursiveNN(vocab, gen_config)
                    elif model_name == 'ResidualRecursiveNN_w_N':
                        from models import ResidualRecursiveNN_w_N
                        model = ResidualRecursiveNN_w_N(vocab, config).cuda()
                        gen_model = ResidualRecursiveNN_w_N(vocab, gen_config)
                    else:
                        raise "No valid model specified: {}".format(model_name) 
                    if model_weights is not None: 
                        print ('Loading pre-trained parameters from {}'.format(model_weights))
                        model.load_state_dict(torch.load(model_weights))
                    train(model, gen_model, model_prefix + model_name, 
                          ctrain, ctest, gtrain, gtest, gval, sc)
