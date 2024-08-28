import networkx as nx
import numpy as np
import pickle as pkl
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from gcn.utils import *
from gcn.models import GCN


import time
import tensorflow as tf

with open('data/gen_intermediate/generalization_pyvex_graphs_train_val_test_split.pkl', 'rb') as f:
    (train, train_labels, val, val_labels, test, test_labels) = pkl.load(f)
del train, val, test

with open('data/gen_intermediate/gcn_format_val.pkl', 'rb') as f:
    (val_adj, val_features, val_one_hot_targets) = pkl.load(f)

    
def convertLabelsForClassification(labels):
    converted_labels = []
    new_labels_mapping = {}
    for l in labels:
        if l not in new_labels_mapping:
            new_labels_mapping[l] = len(new_labels_mapping)
        converted_labels.append(new_labels_mapping[l])
    return np.asarray(converted_labels)

val_labels = convertLabelsForClassification(val_labels)
    
N = len(val_labels)
nb_classes = len(np.unique(val_labels))
val_one_hot_targets = np.zeros((N, nb_classes))
val_one_hot_targets[np.arange(N), val_labels] = 1
print(val_one_hot_targets[0])
    
model_func = GCN
num_supports = 1
batch_size = 64


from gcn.models import GCN
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default() as g:
    with g.name_scope( "ft_graph" ) as g2_scope:
        ft_placeholders = {
            'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, name='features', shape=(None, val_features[0].shape[1])),
            'pooling': tf.sparse_placeholder(tf.float32, name='pooling', shape=(batch_size, None)), 
            'labels': tf.placeholder(tf.float32, name='labels', shape=(None, nb_classes)),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')  # helper variable for sparse dropout
        }
        ft_model = GCN(ft_placeholders, input_dim=val_features[0].shape[1], name="ft_graph", logging=True)
        

num_epochs = 5000
num_shots = 10

def validation_split_for_FT(y, shots=5):
    np.random.seed(2)
    train_idx = []
    for l in np.unique(y):
        train_idx.extend(np.random.choice(np.where(y == l)[0], shots))
    test_idx = np.delete(np.arange(y.shape[-1]), train_idx)
    return train_idx, test_idx

with tf.Session( graph = graph ) as sess:
    sess.run(tf.global_variables_initializer())
    ft_loss, ft_acc = [], []
    ft_loss_val, ft_acc_val = [], []

    # sample the validation training part
    ft_train_idx, ft_test_idx = validation_split_for_FT(val_labels, shots=num_shots)
    ft_train_adj, ft_test_adj = val_adj[ft_train_idx], val_adj[ft_test_idx]
    ft_train_features, ft_test_features = val_features[ft_train_idx], val_features[ft_test_idx]
    ft_train_labels, ft_test_labels = val_one_hot_targets[ft_train_idx], val_one_hot_targets[ft_test_idx]

    # do a few iterations on validation training part
    for epoch in range(num_epochs):
        t = time.time()
        for i, batch in enumerate(construct_batch(ft_train_adj, ft_train_features, ft_train_labels, batch_size)):
            batch_adj_sp, batch_pooling_matrix, batch_features, batch_labels = batch
            batch_features = preprocess_features(batch_features)
            batch_support = [preprocess_adj(batch_adj_sp)]
            batch_pooling_matrix = sparse_to_tuple(batch_pooling_matrix)

            feed_dict = construct_feed_dict(batch_features, 
                                            batch_support, 
                                            batch_labels, 
                                            batch_pooling_matrix,
                                            ft_placeholders, 
                                            batch_size)
            outs = sess.run([ft_model.opt_op, ft_model.loss, ft_model.accuracy], feed_dict=feed_dict)
            ft_loss.append(outs[1])
            ft_acc.append(outs[2])
        # evaluate on validation testing part
        for i, batch in enumerate(construct_batch(ft_test_adj, ft_test_features, ft_test_labels, batch_size)):
            batch_adj_sp, batch_pooling_matrix, batch_features, batch_labels = batch
            batch_features = preprocess_features(batch_features)
            batch_support = [preprocess_adj(batch_adj_sp)]
            batch_pooling_matrix = sparse_to_tuple(batch_pooling_matrix)

            feed_dict = construct_feed_dict(batch_features, 
                                            batch_support, 
                                            batch_labels, 
                                            batch_pooling_matrix,
                                            ft_placeholders, 
                                            batch_size)
            outs = sess.run([ft_model.loss, ft_model.accuracy], feed_dict=feed_dict)
            ft_loss_val.append(outs[0])
            ft_acc_val.append(outs[1])

        print("Epoch:", '%04d' % (epoch + 1), "ft_loss=", "{:.2f}".format(np.mean(ft_loss)),
              "ft_acc=", "{:.2f}".format(np.mean(ft_acc)), 
              "ft_val_loss=", "{:.2f}".format(np.mean(ft_loss_val)),
              "ft_val_acc=", "{:.2f}".format(np.mean(ft_acc_val)), 
              "time=", "{:.2f}".format(time.time() - t))
print("Optimization Finished!")
