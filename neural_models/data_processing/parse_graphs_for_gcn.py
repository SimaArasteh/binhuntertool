import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import sys
from neural_models.tf_gcn.utils import *
from neural_models.tf_gcn.models import Model
from neural_models.tf_gcn import layers
import pickle as pkl
import time
import tensorflow.compat.v1 as tf

class GCN_with_dense(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_with_dense, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.pooling_matrix = placeholders['pooling']
        self.input_dim = input_dim
        self.gcn_out_dim = 64
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        self.build()

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.reprs = self.activations[-1]
        self.pooled_reprs = tf.sparse.sparse_dense_matmul(self.pooling_matrix, self.reprs)
        self.outputs = self.final_dense(self.pooled_reprs)

        # Store model variables for easy access
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in self.variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.variables)
        self.opt_op = self.optimizer.apply_gradients(self.grads_and_vars)

    def build_update_placeholders(self):
        update_placeholders = [tf.placeholder(v.dtype, shape=v.get_shape()) for v in self.variables]
        return update_placeholders

    def build_update_op(self, placeholders):
        self.placeholders['update_placeholders'] = placeholders['update_placeholders']
        self.update_op = [var.assign(pl) for (var, pl) in zip(self.variables, self.placeholders['update_placeholders'])]

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += 5e-4 * tf.nn.l2_loss(var)
        # Cross entropy error
        self.loss += tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.placeholders['labels'], 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        #         self.accuracy = tf.reduce_mean(accuracy_all)
        self.accuracy = accuracy_all

    def _build(self):

        print("\n\n Number of layers changed!: 2->3, number of neurons changed: [32:64:104]\n\n")
        self.layers.append(layers.GraphConvolution(input_dim=self.input_dim,
                                                   output_dim=32,
                                                   placeholders=self.placeholders,
                                                   act=tf.nn.relu,
                                                   dropout=True,
                                                   sparse_inputs=True,
                                                   logging=self.logging))

        self.layers.append(layers.GraphConvolution(input_dim=32,
                                                   output_dim=64,
                                                   placeholders=self.placeholders,
                                                   act=tf.nn.relu,
                                                   dropout=True,
                                                   logging=self.logging))

        self.layers.append(layers.GraphConvolution(input_dim=64,
                                                   output_dim=self.gcn_out_dim,
                                                   placeholders=self.placeholders,
                                                   act=tf.nn.relu,
                                                   dropout=True,
                                                   logging=self.logging))

        self.final_dense = layers.Dense(input_dim=self.gcn_out_dim,
                                        output_dim=self.output_dim,
                                        placeholders=self.placeholders,
                                        act=lambda x: x,
                                        dropout=True)



    

class GCN():

    def __init__(self, graphs=None, labels = None,  train_size=0.7, test_size=0.3, 
                  threshold=5, nb_classes=2, random_seed=43, number_of_epoches = 50, batch_size = 64, number_of_cv = 5):
        """

        """
        self.graphs = graphs
        self.labels = labels
        self.training_data = None
        self.testing_data = None
        self.validation_data = None
        self.train_indexes = None
        self.test_indexes = None
        self.val_indexes = None
        self.threshold = threshold
        self.nb_classes = nb_classes
        self.random_seed = random_seed
        self.number_of_cv = number_of_cv
        self.number_of_epoches = number_of_epoches
        self.batch_size = batch_size
        self.split_data(self.graphs, train_size, test_size, random_seed )

    def split_data(self, data, tr_size, te_size, rand_seed):
        # split graphs to training, testing, validation set
        all_indexes = np.arange(len(data))

        train_indexes, val_and_test_indexes = train_test_split(all_indexes, train_size=tr_size,
                                                            test_size=te_size, random_state=rand_seed)
        val_indexes, test_indexes = train_test_split(val_and_test_indexes, train_size=0.5,
                                                 test_size=0.5, random_state=rand_seed + 1)
        ########################################################
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.val_indexes = val_indexes
        ########################################################
        self.training_data = list(map(data.__getitem__, train_indexes))
        self.testing_data = list(map(data.__getitem__, test_indexes))
        self.validation_data = list(map(data.__getitem__, val_indexes))

    def preprocess(self):
        # create dictionary of words/ create feature matrix, adjacancy matrix, one_hot
        breakpoint()
        node_label_idx_dict, node_label_freq = self.get_node_label_dicts(self.training_data, self.threshold)
        adj, feature, one_hot = self.convert_to_gcn_features(self.graphs, self.labels, node_label_idx_dict, self.nb_classes)

        return adj, feature, one_hot

    def train_gcn(self, adj, feature, one_hot):
        #train the model using GCN
        test_accs_per_cv = []
        val_accs_per_cv = []
        #############
        train_adj = adj[self.train_indexes]
        train_feature = feature[self.train_indexes]
        train_one_hot = one_hot[self.train_indexes]
        #############
        test_adj = adj[self.test_indexes]
        test_feature = feature[self.test_indexes]
        test_one_hot = one_hot[self.test_indexes]
        #############
        val_adj = adj[self.val_indexes]
        val_feature = feature[self.val_indexes]
        val_one_hot = one_hot[self.val_indexes]

        FEATURES_DIM = feature[0][1][0].shape[1]


        # losses_per_cwe = []
        # accs_per_cwe = []

        test_set = False
        num_supports = 1

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default() as g:
            placeholders = {
                'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
                'features': tf.sparse_placeholder(tf.float32, name='features', shape=(None, FEATURES_DIM)),
                'pooling': tf.sparse_placeholder(tf.float32, name='pooling', shape=(self.batch_size, None)),
                'labels': tf.placeholder(tf.float32, name='labels', shape=(None, self.nb_classes)),
                'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
                'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')  # helper variable for sparse dropout
            }
            current_model = GCN_with_dense(placeholders, input_dim=FEATURES_DIM, name="current_graph", logging=True)

        cv_acc = {}
        cv_files = {}
        #file_names = joblib.load('seperate_results_old/CWE121_names.pkl')
        #print("len file names")
        #print(len(file_names))
        for cv in range(self.number_of_cv):
            
            train_data = (train_adj, train_feature, train_one_hot)
            val_data = (val_adj, val_feature, val_one_hot)
            test_data = (test_adj, test_feature, test_one_hot)

            loss, acc, index = self.run_experiment( train_data, val_data, test_data, graph, test_set, current_model, placeholders)


    def run_experiment(self, train_data, test_data, val_data, graph, test_set, current_model, placeholders):
        with tf.Session( graph = graph ) as sess:
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            if not test_set:
                eval_data = val_data
            losses, accs, index = self.regular_experiment(sess, current_model, placeholders,
                                            train_data, eval_data)

            #breakpoint()
            #print(accs)

            print("Test loss = ", "{:.2f}".format(np.mean(losses)),
                "Test acc = ", "{:.2f}".format(np.mean(accs)))
            print ("Completed in {}sec-s".format(int(time.time()-start_time)))
        return np.mean(losses), accs, index

    def run_ops(self, input_ops, sess, idx, data, placeholders):
        adj, features, one_hot_targets = data

        adj = adj[idx]
        features = features[idx]
        labels = one_hot_targets[idx]
        #print("labels are here", labels)
        losses = []
        accuracies = []
        counter = 0
        breakpoint()
        print(adj)
        for batch in construct_batch(adj, features, labels, self.batch_size):
            #breakpoint()
            batch_adj_sp, batch_pooling_matrix, batch_features, batch_labels = batch
            batch_features = preprocess_features(batch_features)
            batch_support = [preprocess_adj(batch_adj_sp)]
            batch_pooling_matrix = sparse_to_tuple(batch_pooling_matrix)
            feed_dict = construct_feed_dict(batch_features,
                                            batch_support,
                                            batch_labels,
                                            batch_pooling_matrix,
                                            placeholders,
                                            self.batch_size)


            outs = sess.run(input_ops, feed_dict=feed_dict)
            #counter+=1
            breakpoint()
            print(outs[1])
            losses.append(outs[0])
            accuracies.extend(outs[1])


    ##################**************************
    # return permutated indexes
        return losses, accuracies, idx


    def regular_experiment(self, sess, model, placeholders, train_data, test_data):
        #breakpoint()
        start_time = time.time()
        train_idx = np.random.permutation(len(train_data[0]))
        test_idx = np.random.permutation(len(test_data[0]))
        breakpoint()
        test_idx = [test_idx[0]]
        print(test_idx)
        print("Len train data: {}, len test data: {}".format(len(train_idx), len(test_idx)))

        l_, a_, index = self.run_ops([model.loss, model.accuracy], sess, test_idx, test_data, placeholders)
        print("Before optimization, loss:{:.2f}, validation acc: {:.2f}".format(np.mean(l_), np.mean(a_)))

        for epoch in range(self.number_of_epoches):
            start_time = time.time()
            l_, a_, index = self.run_ops([model.loss, model.accuracy, model.opt_op], sess, train_idx, train_data, placeholders)
            print("Update {} completed in {:.2f} sec-s".format(epoch, time.time() - start_time),
                "training loss:{:.2f}, training acc: {:.2f}".format(np.mean(l_), np.mean(a_)))
            if epoch != 0 and epoch % 10 == 0:
                l_, a_, index = self.run_ops([model.loss, model.accuracy], sess, test_idx, test_data, placeholders)
                print("Validation loss:{:.2f}, validation acc: {:.2f}".format(np.mean(l_), np.mean(a_)))

        return self.run_ops([model.loss, model.accuracy], sess, test_idx, test_data, placeholders)

     



    def get_node_label_dicts(self, data, threshold):
        #breakpoint()
        node_label_freq = {"UNK": 0}
        for g in data:
            print(type(g.graph))
            node_labels = np.unique([
                v if type(v) is str else "U64" \
                for v in nx.get_node_attributes(g.graph, 'label').values()])
            for l in node_labels:
                l = self.strip_number(l)
                if l not in node_label_freq:
                    node_label_freq[l] = 0
                node_label_freq[l] += 1

        node_label_idx_dict = {"UNK": 0, "CONST": 1}
        #breakpoint()
        for l, f in node_label_freq.items():
            if f >= threshold:
                node_label_idx_dict[l] = len(node_label_idx_dict)
        return node_label_idx_dict, node_label_freq


    def convert_to_gcn_features(self, data, labels,
                                    node_label_idx_dict,
                                    nb_classes):
        
        #breakpoint()
        adj = []
        features = []

        N = len(labels)
        one_hot_targets = np.zeros((N, nb_classes))
        one_hot_targets[np.arange(N), labels] = 1

        E = len(node_label_idx_dict)
        for g in data:
            adj.append(nx.adjacency_matrix(g.graph))
            one_hot_feature = np.zeros((nx.number_of_nodes(g.graph), E))
            node_labels = nx.get_node_attributes(g.graph, 'label')
            for i, n in enumerate(nx.nodes(g.graph)):
                idx = 0
                l = node_labels.get(n, "UNK")
                if n in node_labels:
                    l = node_labels[n]
                    if type(l) is not str:
                        l = "U64"
                    l = self.strip_number(l)
                    if l in node_label_idx_dict:
                        idx = node_label_idx_dict[l]
                    elif l.startswith("0x"):
                        idx = 1
                one_hot_feature[i, idx] = 1
            #breakpoint()
            features.append(sp.csr_matrix(one_hot_feature))
        return np.asarray(adj), np.asarray(features), one_hot_targets


    # only keep last 2 digits of a number
    def get_node_label_dicts_two_digit_numbers(self, data, threshold):
        node_label_freq = {"UNK": 0}
        for g in data:
            node_labels = np.unique([
                v if type(v) is str else "U64" \
                for v in nx.get_node_attributes(g, 'label').values()])
            for l in node_labels:
                l = strip_number(l)
                if is_const(l):
                    char_level = split(l)
                    l = ''.join(char_level[-2:])
                if l not in node_label_freq:
                    node_label_freq[l] = 0
                node_label_freq[l] += 1

        node_label_idx_dict = {"UNK": 0}
        for l, f in node_label_freq.items():
            if f >= threshold:
                node_label_idx_dict[l] = len(node_label_idx_dict)
        return node_label_idx_dict, node_label_freq


    # do not keep any constants, replace them with a term CONST
    def get_node_label_dicts_const_placeholder(self, data, threshold=3):
        node_label_freq = {"UNK": 0}
        for g in data:
            node_labels = np.unique(
                [v if type(v) is str else "U64" \
                for v in nx.get_node_attributes(g, 'label').values()])
            for l in node_labels:
                l = strip_number(l)
                if is_const(l):
                    l = "CONST"
                if l not in node_label_freq:
                    node_label_freq[l] = 0
                node_label_freq[l] += 1

        node_label_idx_dict = {"UNK": 0}
        for l, f in node_label_freq.items():
            if f >= threshold:
                node_label_idx_dict[l] = len(node_label_idx_dict)
        return node_label_idx_dict, node_label_freq


    def old_strip_number(self, l):
        if l.startswith("'reg"):
            l = l.split("_")[0]
        if l.startswith("Source_"):
            l = l.split("_")[0]
        if l.startswith("Sync"):
            l = l.split("_")[0]
        if l.startswith("'t"):
            l = "temp"
        return l


    # in the new version I fixed the bug that led to trailing apostrophes
    def strip_number(self, l):
        if l.startswith("reg"):
            l = l.split("_")[0]
        if l.startswith("Source_"):
            l = l.split("_")[0]
        if l.startswith("Sync"):
            l = l.split("_")[0]
        if l.startswith("t"):
            l = "temp"
        return l


    def is_const(self, k):
        if k.endswith('L'):
            return True
        try:
            int(k)
            return True
        except:
            pass
        try:
            int(k, 16)
            return True
        except:
            pass
        return False


    def convert_to_gcn_features_const_placeholder(self, data, labels,
                                                node_label_idx_dict,
                                                nb_classes=104):
        adj = []
        features = []

        N = len(labels)
        one_hot_targets = np.zeros((N, nb_classes))
        one_hot_targets[np.arange(N), labels] = 1

        E = len(node_label_idx_dict)
        for g in data:
            adj.append(nx.adjacency_matrix(g))
            one_hot_feature = np.zeros((nx.number_of_nodes(g), E))
            node_labels = nx.get_node_attributes(g, 'label')
            for i, n in enumerate(nx.nodes(g)):
                idx = 0
                l = "UNK"
                # if the node has a label:
                if n in node_labels:
                    l = node_labels[n]
                    if type(l) is not str:
                        l = "U64"
                    l = strip_number(l)
                    # if the label is a const, trim to last two digits
                    if is_const(l):
                        l = "CONST"
                if l in node_label_idx_dict:
                    idx = node_label_idx_dict[l]
                one_hot_feature[i, idx] = 1
            features.append(sp.csr_matrix(one_hot_feature))
        return np.asarray(adj), np.asarray(features), one_hot_targets


    def is_num(self, c):
        if c >= '0' and c <= '9':
            return True
        return False


    def is_hex_digit(self, c):
        hex_digits = ['a', 'b', 'c', 'd', 'e', 'f']
        if c.lower() in hex_digits:
            return True
        return False


    def split(self, k):
        res = []
        if k.startswith('0x') or k.startswith('0X'):
            k = k[2:]
        for i in range(len(k)):
            if is_num(k[i]):
                res.append(k[i])
            elif is_hex_digit(k[i]):
                res.append(k[i].lower())
            else:
                if k[i] != 'L':
                    print(k, k[i:])
        return res


    def convert_to_gcn_features_two_digit_numbers(self, data, labels,
                                                node_label_idx_dict,
                                                nb_classes=104):
        adj = []
        features = []

        N = len(labels)
        one_hot_targets = np.zeros((N, nb_classes))
        one_hot_targets[np.arange(N), labels] = 1

        E = len(node_label_idx_dict)
        for g in data:
            adj.append(nx.adjacency_matrix(g))
            one_hot_feature = np.zeros((nx.number_of_nodes(g), E))
            node_labels = nx.get_node_attributes(g, 'label')
            for i, n in enumerate(nx.nodes(g)):
                idx = 0
                l = "UNK"
                # if the node has a label:
                if n in node_labels:
                    l = node_labels[n]
                    if type(l) is not str:
                        l = "U64"
                    l = strip_number(l)
                    # if the label is a const, trim to last two digits
                    if is_const(l):
                        char_level = split(l)
                        l = ''.join(char_level[-2:])
                if l in node_label_idx_dict:
                    idx = node_label_idx_dict[l]
                one_hot_feature[i, idx] = 1
            features.append(sp.csr_matrix(one_hot_feature))
        return np.asarray(adj), np.asarray(features), one_hot_targets
