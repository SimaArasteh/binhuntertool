import numpy as np
import os
import pickle as pkl
import scipy.sparse as sp

DATA_PATH = os.getenv("DATA_PATH")

with open(DATA_PATH + '/updated_graphs/fold_0/old_gcn_on_oj_train.pkl', 'rb') as f:
    train = pkl.load(f)
    train_adj, train_feaat, train_labels = train
with open(DATA_PATH + '/updated_graphs/fold_0/old_gcn_on_oj_test.pkl', 'rb') as f:
    test = pkl.load(f)
    test_adj, test_feat, test_labels = test
with open(DATA_PATH + '/updated_graphs/fold_0/old_gcn_on_oj_val.pkl', 'rb') as f:
    val = pkl.load(f)
    val_adj, val_feat, val_labels = val

train_adj = np.asarray([sp.csr_matrix(a.todense() + np.eye(a.shape[0])) for a in train_adj])
with open(DATA_PATH + '/updated_graphs/fold_0/old_gcn_on_oj_train_w_self_loops.pkl', 'wb') as f:
    pkl.dump(train_adj, f)

val_adj = np.asarray([sp.csr_matrix(a.todense() + np.eye(a.shape[0])) for a in val_adj])
with open(DATA_PATH + '/updated_graphs/fold_0/old_gcn_on_oj_val_w_self_loops.pkl', 'wb') as f:
    pkl.dump(val_adj, f)

test_adj = np.asarray([sp.csr_matrix(a.todense() + np.eye(a.shape[0])) for a in test_adj])
with open(DATA_PATH + '/updated_graphs/fold_0/old_gcn_on_oj_test_w_self_loops.pkl', 'wb') as f:
    pkl.dump(test_adj, f)



