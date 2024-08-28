import bz2
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from sklearn.model_selection import train_test_split

from neural_models.data_processing.parse_graphs_for_gcn import old_get_node_label_dicts
from neural_models.data_processing.parse_graphs_for_gcn import old_convert_to_gcn_features
from neural_models.data_processing.utils import read_all_graphs


class Dataset(object):
    def __init__(self):
        self.splits = None
        self.node_label_dict = None
        self.node_label_freq = None
        self.desc_to_idx = None
        self.data = None
        self.filenumbers = None
        self.labels = None
        self.dataset_description = None
        self.splits = None
        self.train_indexes = None
        self.val_indexes = None
        self.test_indexes = None
        self.random_seed = None
        self.adj = None
        self.feat = None
        self.one_hot_labels = None

    def create_dataset(self, cc, olvl, seed=None, from_dataset=None):
        self.data, self.filenumbers, self.labels = read_all_graphs(cc, olvl)
        self.index_data()
        self.dataset_description = {"cc": cc, "olvl": olvl, "seed": seed}

        if from_dataset is not None:
            self.splits = ()
            for s in from_dataset.splits:
                self_s = []
                for si in s:
                    desc = from_dataset.get_desc(si)
                    if desc in self.desc_to_idx:
                        self_s.append(self.desc_to_idx[desc])
                self_s = np.asarray(self_s)
                self.splits += (self_s,)
            self.train_indexes, self.val_indexes, self.test_indexes = self.splits
        else:
            if seed is not None:
                self.random_seed = seed
            else:
                raise Exception("Random seed for dataset split is not provided!")
            self.split_data()

    def load_dataset(self, filename, compress=False):
        if compress:
            with bz2.BZ2File(filename + '.pbz2', 'r') as f:
                self.__dict__ = pkl.load(f)
        else:
            with open(filename + '.pkl', 'rb') as f:
                self.__dict__ = pkl.load(f)

    def save_dataset(self, filename, compress=False):
        if compress:
            with bz2.BZ2File(filename + '.pbz2', 'w') as f:
                pkl.dump(self.__dict__, f)
        else:
            with open(filename + '.pkl', 'wb') as f:
                pkl.dump(self.__dict__, f)

    def index_data(self):
        self.desc_to_idx = {}
        for i in range(len(self.filenumbers)):
            self.desc_to_idx[self.labels[i], self.filenumbers[i]] = i

    def get_desc(self, i):
        return self.labels[i], self.filenumbers[i]

    def get_data(self, i):
        return self.data[i]

    def split_data(self):
        all_indexes = np.arange(len(self.data))
        self.train_indexes, val_and_test_indexes = train_test_split(all_indexes, train_size=0.7,
                                                                    test_size=0.3, random_state=self.random_seed)
        self.val_indexes, self.test_indexes = train_test_split(val_and_test_indexes, train_size=0.5,
                                                               test_size=0.5, random_state=self.random_seed + 1)

        self.splits = (self.train_indexes, self.val_indexes, self.test_indexes)

    def create_vocab(self):
        assert self.splits, "The data has not been split in training/test/validation"
        self.node_label_dict, self.node_label_freq = old_get_node_label_dicts(
            map(lambda x: self.get_data(x), self.train_indexes))

    def create_adj_and_feat(self):
        self.adj, self.feat, self.one_hot_labels = old_convert_to_gcn_features(self.data, self.labels,
                                                                               self.node_label_dict)

    def add_self_loops(self):
        for i in range(len(self.adj)):
            self.adj[i] = sp.csr_matrix(self.adj[i].todense() + np.eye(self.adj[i].shape[0]))


def main():
    datasets_folder = "/nas/home/shushan/data/shushan/inv_data/datasets/"
    split_0 = Dataset()
    split_0.load_dataset(datasets_folder + "split_0_v2", compress=True)
    print("Split 0 loaded")
    cc = sys.argv[1]
    olvl = sys.argv[2]

    print("Creating dataset for {}, {}".format(cc, olvl))
    d = Dataset()
    d.create_dataset(cc, olvl, from_dataset=split_0)
    print("Dataset created")
    d.create_vocab()
    print("Vocabulary created")
    d.create_adj_and_feat()
    print("Adjacency and feature matrices created")
    d.save_dataset(datasets_folder + "{}_dataset_{}".format(cc, olvl))
    print("Dataset saved to {}{}_dataset_{}.pbz2".format(datasets_folder, cc, olvl))


if __name__ == "__main__":
    main()
