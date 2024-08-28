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


print ("Loading training data...")
with open('/nas/home/shushan/work/data/gen_intermediate/gcn_format_train.pkl', 'rb') as f:
    (train_adj, train_features, train_one_hot_targets) = pkl.load(f)

print ("Done, loading validation data...")   
with open('/nas/home/shushan/work/data/gen_intermediate/gcn_format_val.pkl', 'rb') as f:
    (val_adj, val_features, val_one_hot_targets) = pkl.load(f)

print("Done, starting training")

nb_classes = train_one_hot_targets.shape[-1]
model_func = GCN
num_supports = 1
batch_size = 128

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default() as g:
    with g.name_scope( "train_graph" ) as g1_scope:
        train_placeholders = {
            'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, name='features', shape=(None, train_features[0].shape[1])),
            'pooling': tf.sparse_placeholder(tf.float32, name='pooling', shape=(batch_size, None)), 
            'labels': tf.placeholder(tf.float32, name='labels', shape=(None, nb_classes)),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')  # helper variable for sparse dropout
        }
        model = GCN(train_placeholders, input_dim=train_features[0].shape[1], name="train_graph", logging=True)
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
        
def evaluate(features, support, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)
    
    
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
        
        
        
def validation_split_for_FT(y, shots=5):
    np.random.seed(2)
    train_idx = []
    for l in np.unique(y):
        train_idx.extend(np.random.choice(np.where(y == l)[0], shots))
    test_idx = np.delete(np.arange(y.shape[-1]), train_idx)
    return train_idx, test_idx
    
ft_epochs = 300
num_epochs = 40
meta_epochs = 20

with tf.Session( graph = graph ) as sess:
    sess.run(tf.global_variables_initializer())
    for metaepoch in range(meta_epochs):
        # Train model
        for epoch in range(num_epochs):
            loss = []
            acc = []
            t = time.time()
            for i, batch in enumerate(construct_batch(train_adj, train_features, train_one_hot_targets, batch_size)):
                batch_adj_sp, batch_pooling_matrix, batch_features, batch_labels = batch
                batch_features = preprocess_features(batch_features)
                batch_support = [preprocess_adj(batch_adj_sp)]
                batch_pooling_matrix = sparse_to_tuple(batch_pooling_matrix)

                feed_dict = construct_feed_dict(batch_features, 
                                                batch_support, 
                                                batch_labels, 
                                                batch_pooling_matrix, 
                                                train_placeholders, 
                                                batch_size)
                # Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
                loss.append(outs[1])
                acc.append(outs[2])
            print("Epoch:", '%04d' % (epoch + 1 + 10 * metaepoch), "loss=", "{:.2f}".format(np.mean(loss)),
                  "acc=", "{:.2f}".format(np.mean(acc)),
                  "time=", "{:.2f}".format(time.time() - t))
        ft_loss, ft_acc = [], []
        ft_loss_val, ft_acc_val = [], []
        # copy the parameters
        update_weights = [tf.assign(new, old) for (new, old) in 
           zip(tf.trainable_variables("ft_graph"), tf.trainable_variables("train_graph"))]
        sess.run(update_weights)
        # sample the validation training part
        val_labels = np.argmax(val_one_hot_targets, axis=1)
        ft_train_idx, ft_test_idx = validation_split_for_FT(val_labels, shots=10)
        ft_train_adj, ft_test_adj = val_adj[ft_train_idx], val_adj[ft_test_idx]
        ft_train_features, ft_test_features = val_features[ft_train_idx], val_features[ft_test_idx]
        ft_train_labels, ft_test_labels = val_one_hot_targets[ft_train_idx], val_one_hot_targets[ft_test_idx]
        # do a few iterations on validation training part
        for j in range(ft_epochs):
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

            print("Epoch:", '%04d' % (j + 1), "ft_loss=", "{:.2f}".format(np.mean(ft_loss)),
                  "ft_acc=", "{:.2f}".format(np.mean(ft_acc)), 
                  "ft_val_loss=", "{:.2f}".format(np.mean(ft_loss_val)),
                  "ft_val_acc=", "{:.2f}".format(np.mean(ft_acc_val)), 
                  "time=", "{:.2f}".format(time.time() - t))
print("Optimization Finished!")
