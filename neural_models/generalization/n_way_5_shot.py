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
from gcn.models import GCN, GCNA

    
    
model_func = GCN
num_supports = 1
batch_size = 25
nb_classes = 104
stopping_parameter = 10

def remove_unused_dimensions_from_onehot(one_hot_targets):
    global nb_classes
    
    idx = []
    for i in range(one_hot_targets.shape[1]):
        if np.any(one_hot_targets[:, i]) != 0:
            idx.append(i)
    nb_classes = len(idx)
    new_one_hot_targets = one_hot_targets[:, idx]
    return new_one_hot_targets


def sample_classes_for_n_way(n, one_hot_targets):
    if n < 0 or n > one_hot_targets.shape[1]:
        return np.arange(one_hot_targets.shape[1])
    return np.random.choice(np.arange(one_hot_targets.shape[1]), n, replace=False)
    
    
def few_shot_split(one_hot_targets, shots, n_way):
    sampled_classes = sample_classes_for_n_way(n_way, one_hot_targets)
    print ('sampled classes ', sampled_classes)
    
    train_idx = []
    for l in sampled_classes:
        class_idx = np.where(one_hot_targets[:, l])[0]
        train_idx.extend(np.random.choice(class_idx, shots, replace=False))
    
    one_hot_targets = one_hot_targets[:, sampled_classes]
    sampled_classes_idx = np.where(np.any(one_hot_targets != 0, axis=1))[0]
    
    test_idx = np.delete(sampled_classes_idx, train_idx)
    return train_idx, test_idx


def train(data, model_name, n_way, num_epochs=1000, num_shots=1):
    val_adj, val_features, val_one_hot_targets = data
    
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default() as g:
        with g.name_scope( "ft_graph" ) as g2_scope:
            ft_placeholders = {
                'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
                'features': tf.sparse_placeholder(tf.float32, name='features', shape=(None, val_features[0].shape[1])),
                'pooling': tf.sparse_placeholder(tf.float32, name='pooling', shape=(batch_size, None)), 
                'labels': tf.placeholder(tf.float32, name='labels', shape=(None, n_way)),
                'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
                'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')  # helper variable for sparse dropout
            }
            if model_name == 'GCN':
                ft_model = GCN(ft_placeholders, input_dim=val_features[0].shape[1], name="ft_graph", logging=True)
            elif model_name == 'GCNA':
                ft_model = GCNA(ft_placeholders, input_dim=val_features[0].shape[1], name="ft_graph", logging=True)
                
    with tf.Session( graph = graph ) as sess:
        sess.run(tf.global_variables_initializer())

        avg_loss, avg_acc = [], []
        avg_loss_val, avg_acc_val = [], []

        for epochs in range(num_epochs):
            # sample the validation training part
            ft_train_idx, ft_test_idx = few_shot_split(val_one_hot_targets, shots=num_shots, n_way=n_way)
            ft_train_adj, ft_test_adj = val_adj[ft_train_idx], val_adj[ft_test_idx]
            ft_train_features, ft_test_features = val_features[ft_train_idx], val_features[ft_test_idx]
            ft_train_labels = remove_unused_dimensions_from_onehot(val_one_hot_targets[ft_train_idx])
            ft_test_labels = remove_unused_dimensions_from_onehot(val_one_hot_targets[ft_test_idx])

            # do a few iterations on validation query data points
            e = 0
            ft_loss_val = []
            ft_acc_val = []
#             for it in range(200):
            while True:
                losses = []
                accs = []

                ft_loss, ft_acc = [], []
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
                # evaluate on validation sample data points
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
                    losses.append(outs[0])
                    accs.append(outs[1])

                print("Epoch:", '%04d' % (e + 1), "ft_loss=", "{:.2f}".format(np.mean(ft_loss)),
                      "ft_acc=", "{:.2f}".format(np.mean(ft_acc)), 
                      "ft_val_loss=", "{:.2f}".format(np.mean(losses)),
                      "ft_val_acc=", "{:.2f}".format(np.mean(accs)), 
                      "time=", "{:.2f}".format(time.time() - t))
                ft_loss_val.append(round(np.mean(losses), 2))
                ft_acc_val.append(np.mean(accs))
                e += 1
                if len(ft_loss_val) > stopping_parameter:
                    decreasing_loss = [True for j in range(stopping_parameter, 1, -1) if ft_loss_val[-j] > ft_loss_val[-(j-1)]]
                    if len(decreasing_loss) == 0:
                        break
#                         pass

            avg_loss.append(ft_loss[-1])
            avg_acc.append(ft_acc[-1])
            avg_loss_val.append(ft_loss_val[-1])
            avg_acc_val.append(ft_acc_val[-1])

        print("total average loss = ", "{:.2f}".format(np.mean(avg_loss)),
                  "acc = ", "{:.2f}".format(np.mean(avg_acc)), 
                  "total average val loss = ", "{:.2f}".format(np.mean(avg_loss_val)),
                  "total average val acc = ", "{:.2f}".format(np.mean(avg_acc_val)))
    print("Optimization Finished!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run gcn experiment without fine-tuning, training only on the few query samples from each class.')
    parser.add_argument('--num_shots', type=int, help='Number of query points to use from each class', required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--n_way', type=int, help='Number of classes to test at a time', required=True)
    parser.add_argument('--model_name', type=str, help='Name of the model to use, GCN or GCNA', required=True)
    args = parser.parse_args()
    
    
    with open('/nas/home/shushan/work/data/gen_intermediate/gcn_format_val.pkl', 'rb') as f:
        (val_adj, val_features, val_one_hot_targets) = pkl.load(f)
        
    val_one_hot_targets = remove_unused_dimensions_from_onehot(val_one_hot_targets)

    num_shots = args.num_shots
    n_way = args.n_way
    assert num_shots > 0, "Number of shots (query points) should be positive"
    assert n_way > 0, "Number of classes to test (for N-way testing) should be positive"
    if args.num_epochs:
        num_epochs = args.num_epochs 
    else:
        num_epochs = (25//n_way) * 10
        
    model_name = args.model_name
    known_model_names = ['GCN', 'GCNA']
    assert model_name in known_model_names, 'Unknown model name given'
    
    data = (val_adj, val_features, val_one_hot_targets)
    train(data, model_name, num_shots=num_shots, num_epochs=num_epochs, n_way=n_way)
    
