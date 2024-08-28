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

with open('/nas/home/shushan/work/data/gen_intermediate/gcn_format_train.pkl', 'rb') as f:
    (train_adj, train_features, train_one_hot_targets) = pkl.load(f)
with open('/nas/home/shushan/work/data/gen_intermediate/gcn_format_val.pkl', 'rb') as f:
    (val_adj, val_features, val_one_hot_targets) = pkl.load(f)
    
nb_classes = 104
    
model_func = GCN
num_supports = 1
batch_size = 64

from gcn.models import GCN
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
        saver = tf.train.Saver()

num_epochs = 200

with tf.Session( graph = graph ) as sess:
    sess.run(tf.global_variables_initializer())
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
        print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.2f}".format(np.mean(loss)),
              "acc=", "{:.2f}".format(np.mean(acc)),
              "time=", "{:.2f}".format(time.time() - t))
        if (epoch + 1) % 10 == 0:
            save_path = saver.save(sess, "/nas/home/shushan/work/data/gcn_checkpoints/gcn_epoch_{}.ckpt".format(epoch + 1))
            print("Model saved in path: %s" % save_path)
