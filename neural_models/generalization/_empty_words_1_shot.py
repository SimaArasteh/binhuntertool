from tree import Tree, Node
from sklearn.model_selection import train_test_split
import copy
import pickle as pkl
import numpy as np
import numpy.ma as ma
from vocab import Vocab
from train_gen_exp import train as train_procedure
import torch

with open('data/pyvex_trees.pkl', 'rb') as f:
    trees = pkl.load(f)


def getLabels(data):
    import numpy as np
    
    labels = []
    for d in data:
        labels.append(d.label)
    return labels

labels = getLabels(trees)

def getIndexes(labels, selectLabels):
    idxs = []
    for l in selectLabels:
        idxs.extend(np.argwhere(labels == l))
    idxs = np.asarray(idxs).flatten()
    assert np.array_equal(np.unique(labels[idxs]), np.unique(selectLabels))
    return idxs

def generalizationExperimentSplitData(data, train_size, val_size=None, test_size=None):
    data = np.asarray(data)
    labels = np.asarray(getLabels(data))
    unique_labels = np.unique(labels)
    
    if test_size and val_size:
        assert (train_size + test_size + val_size <= len(unique_labels))
    print("All labels: ", np.unique(labels))
    train_labels, val_and_test_labels = train_test_split(unique_labels, train_size=train_size, random_state=12)
    val_labels, test_labels = train_test_split(val_and_test_labels, train_size=val_size, test_size=test_size, random_state=13)
    print ("{} train labels: {}, \n{} val labels: {}, \n{} test labels: {}".format(
        len(train_labels), train_labels, len(val_labels), val_labels, len(test_labels), test_labels))
    
    train_idx = getIndexes(labels, train_labels)
    train_data = data[train_idx]
    
    val_idx = getIndexes(labels, val_labels)
    val_data = data[val_idx]
    
    test_idx = getIndexes(labels, test_labels)
    test_data = data[test_idx]
    
    return train_data, val_data, test_data

train, val, test = generalizationExperimentSplitData(trees, train_size=50, val_size=25, test_size=None)

vocab = Vocab()
vocab.construct_from_trees(train)

torch.cuda.set_device(1)

# make all labels from 0 to N
def convertDataForClassification(data):
    converted_data = []
    new_labels_mapping = {}
    for dp in data:
        if dp.label not in new_labels_mapping:
            new_labels_mapping[dp.label] = len(new_labels_mapping)
        cdp = copy.copy(dp)
        cdp.label = new_labels_mapping[dp.label]
        converted_data.append(cdp)
    return np.asarray(converted_data)

ctrain, cval, ctest = convertDataForClassification(train), \
                      convertDataForClassification(val), \
                      convertDataForClassification(test)

from models import EmptyWordRecursiveNN, Config, Var

config = Config()
config.label_size = len(np.unique(getLabels(ctrain)))
config.embed_size = 20
config.lr = 1e-3
config.max_epochs = 100

gen_config = copy.deepcopy(config)
gen_config.label_size = len(np.unique(getLabels(cval)))
gen_config.lr = 1e-5
gen_config.max_epochs = 10

model = EmptyWordRecursiveNN(vocab, config).cuda()
gen_model = EmptyWordRecursiveNN(vocab, gen_config).cuda()

train_procedure(model, gen_model, "1_shot_empty_words", ctrain, cval, sc=True, print_every=100, num_shots=1)
