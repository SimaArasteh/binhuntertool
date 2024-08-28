from tree import Tree, Node
from sklearn.model_selection import train_test_split
import copy
import pickle as pkl
import numpy as np
import numpy.ma as ma
from vocab import Vocab
from train_gen_exp import train
import torch


with open('data/trees_v3.pkl', 'rb') as f:
    trees = pkl.load(f)
    
def getLabels(data):
    import numpy as np
    
    labels = []
    for d in data:
        labels.append(d.label)
    return labels
    assert len(np.unique(labels)) == 104

labels = getLabels(trees)


def generalizationExperimentSplitData(data, 
                                      common=51,
                                      common_train_r=0.8, 
                                      common_test_r=0.2,
                                      gen=-1, 
                                      gen_train_count=10,
                                      gen_val_r=0.1
                                     ):
    data = np.asarray(data)
    labels = np.asarray(getLabels(data))
    print("All labels: ", np.unique(labels))
    common_labels = np.arange(common)
    print ("Common labels: ", common_labels)
    if gen > 0:
        gen_labels = np.arange(common, common + gen)
    else:
        gen_labels = np.arange(common, max(np.unique(labels)) + 1)

    common_data = [0 if l in common_labels else 1 for l in labels]
    assert np.all(labels[np.where(not common_data)] < common)
    common_data = ma.masked_array(data, mask=common_data)
    common_data = ma.compressed(common_data)
    
    print("Unique labels of common data: ", np.unique(getLabels(common_data)))
    print("Len of common data: ", len(common_data.data))
    common_train, common_test = train_test_split(common_data, 
                                                 train_size=common_train_r, 
                                                 test_size=common_test_r, 
                                                 random_state=12)
    
    gen_data = [0 if l in gen_labels else 1 for l in labels]
    assert np.all(labels[np.where(not gen_data)] >= common)
    if gen > 0:
        assert np.all(labels[np.where(not gen_data)] < gen + common)
    gen_data = ma.masked_array(data, mask=gen_data)
    gen_data = ma.compressed(gen_data)
    print("Len of gen data: ", len(gen_data))
    
    gen_train = []
    gen_val = []
    gen_test = []
    for l in gen_labels:
        gen_data_for_one_label = []
        for d in gen_data:
            if d.label == l:
                gen_data_for_one_label.append(d)
#         print ("Len of gen_data for label {}: {}".format(l, 
#                                                          len(gen_data_for_one_label)))
                
        gen_train_l, gen_val_and_test_l = train_test_split(gen_data_for_one_label, 
                                                           train_size=gen_train_count, 
                                                           random_state=13)
        
        gen_val_l, gen_test_l = train_test_split(gen_val_and_test_l, 
                                                 train_size=gen_val_r, 
                                                 random_state=14)
        gen_train.extend(gen_train_l)
        gen_val.extend(gen_val_l)
        gen_test.extend(gen_test_l)
    print ("Size of common train: ", len(common_train))
    print ("Size of common test: ", len(common_test))
    print ("Size of gen train: ", len(gen_train))
    print ("Size of gen val: ", len(gen_val))
    print ("Size of gen test: ", len(gen_test))
    return common_train, common_test, np.asarray(gen_train), np.asarray(gen_val), np.asarray(gen_test)
    
    
ctrain, ctest, gtrain, gval, gtest = generalizationExperimentSplitData(trees)

vocab = Vocab()
vocab.construct_from_trees(ctrain)


# ctrain = ctrain[:100]
# ctest = ctest[:100]
clabel_size = len(np.unique(getLabels(ctrain)))
glabel_size = len(np.unique(getLabels(gtrain)))

print(clabel_size, glabel_size)

torch.cuda.set_device(1)

from models import RecursiveNN, Config, Var

config = Config()
config.label_size = clabel_size
gen_config = copy.deepcopy(config)
# gen_config.label_size = clabel_size + glabel_size
gen_config.label_size = 104

model = RecursiveNN(vocab, config).cuda()
gen_model = RecursiveNN(vocab, gen_config).cuda()

train(model, gen_model, "Gen_experiment", ctrain, ctest, gtrain, gval, gtest, sc=False, print_every=5000)