import datetime
import itertools
import numpy as np
import os
import sys
import time
import torch
import random

from tree import Tree
from utils import splitData
from models import Config, Var

from sklearn.model_selection import GroupShuffleSplit
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

def get_episode(x, y=None, shots=5):
    gss = GroupShuffleSplit(n_splits=5, test_size=10)
    splits = list(gss.split(x, groups=y))
    for train_idx, test_idx in splits:
        # now need to take few examples from each class in test and add them to train
        unique_test_labels = np.unique(y[test_idx])
        for l in unique_test_labels:
            class_l_idxs = np.where(y == l)[0]            
            add_to_train_idx = np.random.choice(class_l_idxs, shots)
            train_idx = np.concatenate((train_idx, add_to_train_idx))
            remove_from_test_idx = []
            for i, value in enumerate(test_idx):
                if value in add_to_train_idx:
                    remove_from_test_idx.append(i)
            test_idx = np.delete(test_idx, remove_from_test_idx)
        yield train_idx, test_idx
    


def fine_tune(model, gen_model, val, num_shots, print_every=5):
    optimizer = torch.optim.RMSprop(list(filter(lambda p: p.requires_grad, gen_model.parameters())), 
                                    lr=gen_model.config.lr)
    train_acc = []
    test_acc = []
    for train, test in get_episode(val, shots=5):
        gen_model.set_weights_for_gen_exp(model)
        for ft_epoch in gen_model.config.max_epochs:
            _, _ = run_epoch(gen_model, optimizer, train)
            if ft_epoch % print_every == 0:
                report_result(gen_model, test, "Fine tuning epoch {}".format(ft_epoch), p=-1)
                
        train_acc.append(report_result(gen_model, gtrain, "After fine tuning train", p=-1)[0])
        test_acc.append(report_result(gen_model, val, "After fine tuning val", p=-1)[0])
    print ("Average train acc after fine tuning {}".format(mean(train_acc)))
    print ("Average test acc after fine tuning {}".format(mean(test_acc)))
    return train_loss_history, val_loss_history
        

def train(model, gen_model, model_prefix, train, val, sc=False, print_every=5000, num_shots=5):
    bestAcc = 0
    
    train_loss_history = []
    val_loss_history = []
    
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
        _, _ = run_epoch(model, optimizer, train, print_every)
        end_time = time.time()
        print("Epoch completed in {} mins".format((end_time - start_time)//60))
        sys.stdout.flush()
        
        acc, loss = report_result(model, train, "Train", p=1000)
        train_loss_history.append(loss)
        
        gtrain_loss_history_i, gval_loss_history_i = fine_tune(model, gen_model, val, num_shots)
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
