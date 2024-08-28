import argparse
import networkx as nx
import numpy as np
import pickle as pkl
import sklearn 
import time
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from gcn.utils import *
from gcn.models import GCN

model_func = GCN
num_supports = 1
batch_size = 64
nb_classes = 104


def remove_unused_dimensions_from_onehot(one_hot_targets):
    global nb_classes
    
    idx = []
    for i in range(nb_classes):
        if np.any(one_hot_targets[:, i]) != 0:
            idx.append(i)
    nb_classes = len(idx)
    new_one_hot_targets = one_hot_targets[:, idx]
    return new_one_hot_targets


def load_data_for_gcn():
    with open('data/gen_intermediate/gcn_format_val.pkl', 'rb') as f:
        (val_adj, val_features, val_one_hot_targets) = pkl.load(f)
    val_one_hot_targets = remove_unused_dimensions_from_onehot(val_one_hot_targets)
    return val_adj, val_features, val_one_hot_targets


def few_shot_split(y, shots=5):
    np.random.seed(2)
    train_idx = []
    for l in np.unique(y):
        train_idx.extend(np.random.choice(np.where(y == l)[0], shots))
    test_idx = np.delete(np.arange(y.shape[-1]), train_idx)
    return train_idx, test_idx


def train(data, num_epochs=1000, num_shots=1):
    val_adj, val_features, val_one_hot_targets = data
    
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


    with tf.Session( graph = graph ) as sess:
        sess.run(tf.global_variables_initializer())
        ft_loss, ft_acc = [], []
        ft_loss_val, ft_acc_val = [], []

        # sample the validation training part
        ft_train_idx, ft_test_idx = few_shot_split(val_labels, shots=num_shots)
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
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run gcn experiment without fine-tuning, training only on the few query samples from each class.')
    parser.add_argument('--num_shots', type=int, help='Number of query points to use from each class', required=True)
    parser.add_argument('--num_epochs', type=str, help='Number of epochs to train')
    args = parser.parse_args()
    
    num_shots = args.num_shots
    assert num_shots > 0, "Number of shots (query points) should be positive"
    if args.num_epochs:
        num_epochs = args.num_epochs 
    else:
        num_epochs = 250000//(25*num_shots)
    
    data = load_data_for_gcn()
    train(data, num_shots=num_shots, num_epochs=num_epochs)
    
    