
"""
    This module train and test the classifieris using GCN Model
"""
import numpy as np
import math
import itertools
import random
import networkx as nx
from tabulate import tabulate
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# these are imported from kipf module
from neural_models.tf_gcn.utils import *
from neural_models.tf_gcn.models import Model
from neural_models.tf_gcn import layers
######################################
from elftools.elf.elffile import ELFFile
import pickle as pkl
import time 
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from collections import Counter
import joblib
import logging
from sklearn.metrics import precision_score

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
    def __init__(self, graphs, pdg_graphs, cwe_type,  labels, bin_paths,  params, allowed_edges,real_world_info,  train_size=0.7, test_size=0.3, nb_classes=2, 
    random_seed=43,    batch_size = 64, number_of_cv = 5 ):

        self.graphs = graphs
        self.cwe_type = cwe_type
        self.pdgs = pdg_graphs
        self.labels = labels
        self.binary_paths = bin_paths
        self.training_data = None
        self.testing_data = None
        self.realworld_info = real_world_info
        self.validation_data = None
        self.train_indexes = None
        self.test_indexes = None
        self.val_indexes = None
        self.word_dictionary = None
        self.average_bb_number = None
        self.params = params
        #self.function_names = function_names
        self.allowed_edges = allowed_edges
        self.nb_classes = nb_classes 
        self.random_seed = random_seed
        self.number_of_cv = number_of_cv
        self.number_of_epoches = params['epochs']
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
        #for training graphs use debug section to find the exact portion 
        self.training_data = list(map(data.__getitem__, self.train_indexes))
        # for test and validation we do not have access to the debug section
        self.testing_data = list(map(self.pdgs.__getitem__, self.test_indexes))
        self.validation_data = list(map(self.pdgs.__getitem__, self.val_indexes))
        #####
        self.testing_data_db = list(map(data.__getitem__, self.test_indexes))
        self.validation_data_db = list(map(data.__getitem__, self.val_indexes))
        
        

    
    def find_all_call_to_lib(self):
        all_calls = []
        training_binaries = np.array(self.binary_paths)[self.train_indexes]
        test_binaries = np.array(self.binary_paths)[self.test_indexes]
        valid_binaries = np.array(self.binary_paths)[self.val_indexes]
        all_bins = list(training_binaries)+list(test_binaries)+list(valid_binaries)
        all_data = list(self.training_data)+list(self.testing_data)+list(self.validation_data)
        for idx in range(len(all_bins)):
            subg = all_data[idx]
            binp = all_bins[idx]
            all_calls.append(self.find_call_to_api_lib(binp, subg))

        return all_calls


    
    
    def find_call_to_api_lib(self, bin_path, subgraph):
        # This function verifies if a subgraph contains any call to library/API function
        proj = angr.Project(bin_path, load_options={'auto_load_libs': False})
        cfg = proj.analyses.CFGFast()
        range_subgraph_address = subgraph.graph['range_of_address']
        #print(range_subgraph_address)
        call_to_library = set()
        with open(bin_path, 'rb') as f:
            elf = ELFFile(f)

            # Get the section containing the dynamic symbol table
            dynsym_section = elf.get_section_by_name('.dynsym')

            # Get the addresses of all imported symbols

            for symbol in dynsym_section.iter_symbols():
                if symbol['st_info']['bind'] == 'STB_GLOBAL' and symbol['st_shndx'] == 'SHN_UNDEF':
                    obj = proj.loader.main_object
                    base = obj.min_addr
                    if symbol.name in list(obj.plt.keys()):
                        func_lib_addr = obj.plt[symbol.name]
                        for node in cfg.graph.nodes():
                            if node.block is not None:
                                for ins in node.block.capstone.insns:

                                    offset = ins.address-base

                                    full_adr = hex(1048576 +offset)
                                    if full_adr in range_subgraph_address and ins.mnemonic == 'call':
                                    
                                        target = ins.op_str
                                        if len(target.split(" ")) == 1:
                                            if target.startswith("0x"):
                                                if target == hex(func_lib_addr):
                                                    call_to_library.add(symbol.name)


        return list(call_to_library)

    
    def get_node_label_dicts(self):
        # we consider three types of nodes 
        # const, operation and register
        #address and Unique are ignored because unique is a temporary variable and
        #extract aforementioned node labels from the training graphs
        #valid_label_types = ['Register','Const', 'Operation','Unique','Address']
        unique_labels_of_graphs = set()
        labels_with_indexes = {}
        index = 0

        for g in self.training_data:
            

            g_labels = nx.get_node_attributes(g, 'label')
            g_types = nx.get_node_attributes(g, 'node_type')

            for l in list(g_labels.values()):
                node_value = list(g_labels.keys())[list(g_labels.values()).index(l)]
                node_type = g_types[node_value]

                unique_labels_of_graphs.add(l)

            

        for l in unique_labels_of_graphs:
            labels_with_indexes[l] = index
            index+=1

        return unique_labels_of_graphs, labels_with_indexes
    

    def convert_to_gcn_features(self, labels_with_indexes):
        adj = []
        features = []
        training_labels = np.array(self.labels)[self.train_indexes]
        N = len(training_labels)
        one_hot_targets = np.zeros((N, self.nb_classes))
        one_hot_targets[np.arange(N), training_labels] = 1

        E = len(labels_with_indexes)
        for g in self.training_data:
            adj.append(nx.adjacency_matrix(g))
            one_hot_feature = np.zeros((nx.number_of_nodes(g), E))

            for node, node_attr in g.nodes(data=True):
                label = node_attr['label']
                node_idx = list(g.nodes()).index(node)
                if label in list(labels_with_indexes.keys()):
                    number = labels_with_indexes[label]
                    one_hot_feature[node_idx, number] = 1
            
            features.append(sp.csr_matrix(one_hot_feature))
        return np.asarray(adj), np.asarray(features), one_hot_targets

    def convert_to_gcn_features_new(self, g, label_r):
        adj = []
        features = []
        one_hot_targets = np.zeros((1, self.nb_classes))
        one_hot_targets[np.arange(1), label_r] = 1
        if self.word_dictionary is None:
            self.word_dictionary = joblib.load('saved_trained_model/'+self.cwe_type+'/word_dict.pkl')
            

        E = len(self.word_dictionary)
        one_adj = nx.adjacency_matrix(g)
        one_hot_feature = np.zeros((nx.number_of_nodes(g), E))

        for node, node_attr in g.nodes(data=True):
            label = node_attr['label']
            node_idx = list(g.nodes()).index(node)
            if label in list(self.word_dictionary.keys()):
                number = self.word_dictionary[label]
                one_hot_feature[node_idx, number] = 1
        

        return one_adj, one_hot_feature, one_hot_targets


        
    def convert_to_gcn_features_db_testing(self):
        
        testing_labels = np.array(self.labels)[self.test_indexes]
        validation_labels = np.array(self.labels)[self.val_indexes]
        ######################for testing data ##################################
        
        adj = []
        features = []
        N = len(testing_labels)
        one_hot_targets = np.zeros((N, self.nb_classes))
        one_hot_targets[np.arange(N), testing_labels] = 1
        for g in self.testing_data_db:
            g_label = testing_labels[self.testing_data_db.index(g)]
            adjaxancy, one_hot_feature, one_hot = self.convert_to_gcn_features_new(g, g_label)
            adj.append(adjaxancy)
            features.append(sp.csr_matrix(one_hot_feature))
        
        ######################for validation data #########################
        adj_val = []
        features_val = []
        N2 = len(validation_labels)
        one_hot_targets_val = np.zeros((N2, self.nb_classes))
        one_hot_targets_val[np.arange(N2), validation_labels] = 1
        for g_val in self.validation_data_db:
            g_label_val = validation_labels[self.validation_data_db.index(g_val)]
            adjaxancy_val, one_hot_feature_val, one_hot_val = self.convert_to_gcn_features_new(g_val, g_label_val)
            adj_val.append(adjaxancy_val)
            features_val.append(sp.csr_matrix(one_hot_feature_val))
        
        return [np.asarray(adj), np.asarray(features), one_hot_targets], [np.asarray(adj_val), np.asarray(features_val), one_hot_targets_val ]



        
    def compute_over(self, set_slice, set_ground):
        slice_offsets = set()
        for item in set_slice:
            offset = hex(int(item,16)-1048576)
            slice_offsets.add(offset)
        com = list(slice_offsets & set(set_ground))
        rate = len(com)/len(set_ground)
        return rate


    def convert_to_gcn_features_testing(self):

 
        avg_num = self.compute_avg_bbs_in_training_graphs()
        avg_bb_train = math.ceil(avg_num)

        self.average_bb_number = avg_bb_train
        testing_adj = []
        testing_feature = []
        testing_one_hot = []
        testing_size = []
        testing_labels = np.array(self.labels)[self.test_indexes]
        testing_binaries = np.array(self.binary_paths)[self.test_indexes]
        testing_permmitted_edges = np.array(self.allowed_edges)[self.test_indexes]
        testing_subgraphs = list(map(self.graphs.__getitem__, self.test_indexes))
        #########################################################################
        validation_labels = np.array(self.labels)[self.val_indexes]
        validation_binaries = np.array(self.binary_paths)[self.val_indexes]
        validation_permmitted_edges = np.array(self.allowed_edges)[self.val_indexes]
        validation_subgraphs = list(map(self.graphs.__getitem__, self.val_indexes))
        ######################for testing data ##################################
        g_labels = []
        graph_real_addresses = []
        graph_test_range_address = []
        g_paths = []
        total_number_slices = 0
        evaluation_start_time = time.time()
        for g in self.testing_data:
            adj_slices = []
            feature_slices = []
            g_label = testing_labels[self.testing_data.index(g)]
            g_path = testing_binaries[self.testing_data.index(g)]
            g_permmitted_edge = testing_permmitted_edges[self.testing_data.index(g)]
            testing_subgraph = testing_subgraphs[self.testing_data.index(g)]

            g_slices, range_test_slice , o_sizes= self.slice_pdg_testset(g,avg_bb_train, g_permmitted_edge)

            total_number_slices = total_number_slices+ len(g_slices)

            if len(g_slices):
                for idx in range(len(g_slices)):
                    slic = g_slices[idx]

                    adj, feat, one_hot =  self.convert_to_gcn_features_new(slic, g_label)
                    adj_slices.append(adj)
                    feature_slices.append(sp.csr_matrix(feat))

                testing_adj.append(adj_slices)
                testing_feature.append(feature_slices)
                g_labels.append(g_label)
                testing_size.append(o_sizes)
                graph_real_addresses.append(testing_subgraph.graph['debug_line_addressess'])
                graph_test_range_address.append(range_test_slice)

                g_paths.append(g_path)

        
        evaluation_time_part1 = time.time()-evaluation_start_time

        N = len(g_labels)
        one_hot_targets = np.zeros((N, self.nb_classes))
        one_hot_targets[np.arange(N), g_labels] = 1
        ###############################for validation data###########################
        validation_adj = []
        validation_feature = []
        validation_one_hot = []
        validation_sizes = []
        gra_labels = []
        graph_val_real_addr = []
        graph_val_slice_addr = []
        g_val_paths = []
        
        for gra in self.validation_data:
            adj_slices_val = []
            feature_slices_val = []
            g_label_val = validation_labels[self.validation_data.index(gra)]
            g_path_val = validation_binaries[self.validation_data.index(gra)]
            g_permmitted_edge_val = validation_permmitted_edges[self.validation_data.index(gra)]
            subgraph_val = validation_subgraphs[self.validation_data.index(gra)]

            g_slices_val, range_adress_slices , v_sizes= self.slice_pdg_testset(gra,avg_bb_train, g_permmitted_edge_val)

            if len(g_slices_val):
                for jdx in range(len(g_slices_val)):
                    sli = g_slices_val[jdx]

                    adj_val, feat_val, one_hot_val =  self.convert_to_gcn_features_new(sli, g_label_val)
                    adj_slices_val.append(adj_val)
                    feature_slices_val.append(sp.csr_matrix(feat_val))


                validation_adj.append(adj_slices)
                validation_feature.append(feature_slices)
                validation_sizes.append(v_sizes)
                gra_labels.append(g_label_val)
                graph_val_real_addr.append(subgraph_val.graph['debug_line_addressess'])
                graph_val_slice_addr.append(range_adress_slices)
                g_val_paths.append(g_path_val)

        N = len(gra_labels)
        one_hot_targets_val = np.zeros((N, self.nb_classes))
        one_hot_targets_val[np.arange(N), gra_labels] = 1

        return [testing_adj, testing_feature, one_hot_targets,testing_size, graph_real_addresses, graph_test_range_address, g_paths],[validation_adj, validation_feature,one_hot_targets_val,validation_sizes, graph_val_real_addr, graph_val_slice_addr, g_val_paths] , total_number_slices
        

    def preprocess(self):
        unique_labels , node_indexes = self.get_node_label_dicts()
        self.word_dictionary = node_indexes

        training_labels = np.array(self.labels)[self.train_indexes]
        adj = []
        features = []
        N = len(training_labels)
        one_hot_targets = np.zeros((N, self.nb_classes))
        one_hot_targets[np.arange(N), training_labels] = 1
        for g in self.training_data:
            
            g_label = training_labels[self.training_data.index(g)]

            adjaxancy, one_hot_feature, one_hot = self.convert_to_gcn_features_new(g, g_label)
            adj.append(adjaxancy)
            features.append(sp.csr_matrix(one_hot_feature))
        
        
        return np.asarray(adj), np.asarray(features), one_hot_targets
    
    def compute_avg_bbs_in_training_graphs(self):

        sum_numbers = 0
        avg_bbs = 0
        count = 0
        training_labels = np.array(self.labels)[self.train_indexes]
        training_bin_paths = np.array(self.binary_paths)[self.train_indexes]
        for g in self.training_data:
            g_label = training_labels[self.training_data.index(g)]
            range_addr = g.graph['range_of_address']
            bin_path = training_bin_paths[self.training_data.index(g)]

            if g_label == 1:
                #consider average only for vulnerable ones
                sum_numbers = sum_numbers+len(list(set(range_addr)))
                count+=1

        if count!=0:
            avg = sum_numbers/count
        
            return avg
        else:
                return None
        


    def find_function_call(self, graph):
        address_of_function_calls = set()

        for node, node_attr in graph.nodes(data=True):
            
            if node == 'Operation_CALL':
                address_of_function_calls.update(node_attr['ins_addr'])

        return address_of_function_calls

    
    def slice_pdg_testset(self, ddg, avg_num, permmited_edgs):

        slices = []
        range_address = []
        over_size = []

        address_function_call = self.find_function_call(ddg)
        #slice based on function calls and average number
        addres_sort = sorted(address_function_call)
        for adr in addres_sort:

            slice_ins = self.find_slice_address(adr, avg_num, permmited_edgs)

            want = list(set(slice_ins))
            want.sort()

            range_address.append(want)


            subgraph_test = self.create_one_slice(list(set(slice_ins)), ddg, permmited_edgs)

            slices.append(subgraph_test)
            size_sub = subgraph_test.size()
            size_orig = ddg.size()
            over_size.append(size_sub/size_orig)
        

        assert len(slices) == len(range_address) == len(over_size)




        return slices  , range_address, over_size

    def find_slice_address(self, adres, avg, edges):
        all_ins = list(edges.keys())
        all_ins.sort()

        ins_index = all_ins.index(adres)
        start = ins_index-avg
        end = ins_index+avg
        if start < 0:
            start = 0
        if end > len(all_ins):
            end = len(all_ins)
        

        return all_ins[start:end]

    def create_one_slice(self, slice_addresses, ddg_g , permit_edges):

        
        list_nodes = []
        slice_edges = []
        for addres in slice_addresses:
            slice_edges = slice_edges + permit_edges[addres]
            for no , no_attr in ddg_g.nodes(data=True):
                if addres in no_attr['ins_addr'] and no is not None:
                    list_nodes.append(no)

        subgraph_test = ddg_g.subgraph(list_nodes)
        bad_edges = []
        for edge in subgraph_test.edges():
            if edge not in slice_edges:
                bad_edges.append(edge)
        

        subgraph_test_copy = subgraph_test.copy()
        subgraph_test_copy.remove_edges_from(bad_edges)

        return subgraph_test_copy



    def test_on_realworld(self, realworld_ddg, realworld_subgraph, realworld_label, bin_path, function_name):
        adj = []
        features = []
        one_hot_targets = np.zeros((1, self.nb_classes))
        one_hot_targets[np.arange(1), realworld_label] = 1
        if self.word_dictionary is None:
            self.word_dictionary = joblib.load('saved_trained_model/'+self.cwe_type+'/word_dict.pkl')
            

        E = len(self.word_dictionary)
        one_adj = nx.adjacency_matrix(realworld_subgraph)
        one_hot_feature = np.zeros((nx.number_of_nodes(realworld_subgraph), E))

        for node, node_attr in realworld_subgraph.nodes(data=True):
            label = node_attr['label']
            node_idx = list(realworld_subgraph.nodes()).index(node)
            if label in list(self.word_dictionary.keys()):
                number = self.word_dictionary[label]
                one_hot_feature[node_idx, number] = 1
        

        return np.asarray([one_adj]), np.asarray([sp.csr_matrix(one_hot_feature)]), one_hot_targets

        



        

    def train_gcn(self):
        train_adj , train_feature , train_one_hot = self.preprocess()
        testing, validation , number_slices = self.convert_to_gcn_features_testing()
        testing_db, validation_db = self.convert_to_gcn_features_db_testing()
        test_accs_per_cv = []
        val_accs_per_cv = []
    
        #############
        test_adj = testing[0]
        test_feature = testing[1]
        test_one_hot = testing[2]
        ov_sizes = testing[3]
        test_real_address = testing[4]
        test_slices_address = testing[5]
        test_bin_paths = testing[6]
        #############
        val_adj = validation[0]
        val_feature = validation[1]
        val_one_hot = validation[2]
        v_sizes = validation[3]
        val_real_address = validation[4]
        val_slices_addresses = validation[5]
        val_bin_paths = validation[6]

        ##############

        test_adj_db = testing_db[0]
        test_feature_db = testing_db[1]
        test_one_hot_hot_db = testing_db[2]
        ##############

        val_adj_db = validation_db[0]
        val_feature_db = validation_db[1]
        val_one_hot_db = validation_db[2]
        #############
        

        FEATURES_DIM = len(self.word_dictionary)

        


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

        
        tpr_kol = []
        fpr_kol = []
        for cv in range(self.number_of_cv):
            
            train_data = (train_adj, train_feature, train_one_hot)
            val_data = (val_adj, val_feature, val_one_hot, v_sizes,  val_real_address, val_slices_addresses, val_bin_paths)
            test_data = (test_adj, test_feature, test_one_hot, ov_sizes, test_real_address, test_slices_address, test_bin_paths)
            db_testig_data = (test_adj_db, test_feature_db,test_one_hot_hot_db )
            db_validation_data = (val_adj_db, val_feature_db,val_one_hot_db )
            
            rea , dbb  = self.run_experiment( train_data,  test_data, val_data, graph, test_set, current_model, placeholders, db_testig_data, db_validation_data, cv)
            cv_acc[cv] = (rea, dbb)
            tpr_kol.append(rea[4])
            fpr_kol.append(rea[5])
        
        table_data = [
            
            ["Total number of slices", number_slices, "", "", ""],
            ["True Positive Rate", "{:.2f}".format(np.mean(tpr_kol)), "False Positive Rate", "{:.2f}".format(np.mean(fpr_kol)), "", ""]
        ]
        #breakpoint()
        headers = ["Measure", "Value", "Measure", "Value", "Measure", "Value"]
        column_alignment = ("left", "right", "left", "right", "left", "right")
        print(tabulate(table_data, headers=headers, tablefmt="grid", colalign=column_alignment))

        
        
    
    def run_experiment(self, train_data, test_data, val_data, graph, test_set, current_model, placeholders, db_tesing, db_validation, cv):
        with tf.Session( graph = graph ) as sess:
            start_time = time.time()
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            real, db = self.regular_experiment(sess, current_model, placeholders,train_data, test_data, val_data, db_tesing, db_validation)
            
            
            print ("Evaluation Completed in {}sec-s".format(int(time.time()-start_time)))
            
            
        return [real[0], np.mean(real[1]), real[2], real[3], real[4], real[5]], [np.mean(db[0]), np.mean(db[1]), db[2]]

    def run_ops(self, input_ops, sess, idx, data, placeholders, train_flg):

        if train_flg :
            

            adj, features, one_hot_targets = data

            adj = adj[idx]
            features = features[idx]
            labels = one_hot_targets[idx]
            normal_labels = np.argmax(labels, axis=1)

            losses = []
            accuracies = []
            predicted_ls = []
            counter = 0
            

            for batch in construct_batch(adj, features, labels, self.batch_size):

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

                losses.append(outs[0])
                accuracies.extend(outs[1])


            for idx in range(len(accuracies)):
                if accuracies[idx] == 1:
                    predicted_ls.append(normal_labels[idx])
                else:
                    pr_l = int (not normal_labels[idx])
                    predicted_ls.append(pr_l)


            
            return losses, accuracies, idx
        else:
            

            adj_testing, features_testing, one_hot_targets_testing ,s_sizesss, real_adresses, slice_adress, test_bin_pathes= data

            adj_testing_new = []
            features_testing_new = []
            one_hot_targets_testing_new = []
            real_adresses_new = []
            slice_adress_new = []
            test_bin_pathes_new = []
            new_sizes = []
            for item in idx:
                adj_testing_new.append(adj_testing[item])
                features_testing_new.append(features_testing[item])
                one_hot_targets_testing_new.append(one_hot_targets_testing[item])
                real_adresses_new.append(real_adresses[item])
                slice_adress_new.append(slice_adress[item])
                test_bin_pathes_new.append(test_bin_pathes[item])
                new_sizes.append(s_sizesss[item])




            

            losses = []
            acc_for_all_testing = []
            predicted_labels = []
            number_true_positives = 0
            number_true_negatives = 0
            number_false_postives = 0
            number_false_negatives = 0

            sum_kol_overlap = 0
            sum_kol_local = 0
            count_overlap = 0
            overs_kol = []
            acc_kol = []
            t_labels = []

            for jdx in range(len(adj_testing_new)):

                overs = []
                accuracies = []
                len_slices = []
                sizes = []
                list_slices = adj_testing_new[jdx]
                siz_over_slices = new_sizes[jdx]
                
                labels = [one_hot_targets_testing_new[jdx]]
                address_real_graph = real_adresses_new[jdx]
                slice_adresses_for_graph = slice_adress_new[jdx]
                bin_pa = test_bin_pathes_new[jdx]

                for idx_ in range(len(list_slices)):

                
                    try:
                        adj_slice = np.asarray([list_slices[idx_]])
                        feature_slice = np.asarray([features_testing_new[jdx][idx_]])
                        addres_for_one_slice = slice_adresses_for_graph[idx_]
                        over = self.compute_over(addres_for_one_slice, address_real_graph)
                        overs.append(over)
                        overs_kol.append(over)
                        sizes.append(siz_over_slices[idx_])

                        for batch in construct_batch(adj_slice, feature_slice, labels, self.batch_size):

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

                            accuracies.append(outs[1][0])
                            acc_kol.append(outs[1][0])
                            
                    except:
                        continue 
                
                labels_index_zero = labels[0][0]
                labels_index_one = labels[0][1]

                if labels_index_one and self.check_in_size(sizes) :
                    
                    
                    
                    

                    if any(accuracies):
                        

                        
                        number_true_positives+=1
                        predicted_labels.append(1)
                        t_labels.append(1)

                        acc_for_all_testing.append(1)
                    else:

                        number_false_negatives+=1
                        predicted_labels.append(0)
                        acc_for_all_testing.append(0)
                        t_labels.append(1)


                    

                if labels_index_zero : 

                    c = Counter(accuracies)
                    value, count = c.most_common()[0]

                    if any(accuracies) and self.check_over(overs):
                        
                        number_true_negatives+=1
                        predicted_labels.append(0)
                        acc_for_all_testing.append(1)
                        t_labels.append(0)
                    else:
                        
                        number_false_postives+=1
                        predicted_labels.append(1)
                        acc_for_all_testing.append(0)
                        t_labels.append(0)
                
                

            FPR = 0
            TPR = 0
            if number_true_positives+number_false_negatives :
                TPR = number_true_positives/ (number_true_positives+number_false_negatives)
                
            if number_false_postives+number_true_negatives:
                FPR = number_false_postives/ (number_false_postives+number_true_negatives)

            return losses, acc_for_all_testing, idx, 1, TPR, FPR


    def check_over(self, ls_overs):
        for element in ls_overs:
            if element < self.params['over_rate']:
                return False
        return True
    def check_in_size(self, list_sizes):
        for item in list_sizes:
            if item >self.params['over_size']:
                return False
        return True

    def regular_experiment(self, sess, model, placeholders, train_data, test_data, val_data,  db_test, db_val):
        
        start_time = time.time()
        train_idx = np.random.permutation(len(train_data[0]))
        test_idx = np.random.permutation(len(test_data[0]))
        val_idx = np.random.permutation(len(val_data[0]))
        test_idx_db = np.random.permutation(len(db_test[0]))
        val_idx_db = np.random.permutation(len(db_val[0]))

        train_flag = False
        
        

        
        for epoch in range(self.number_of_epoches):
            start_time = time.time()
            train_flag = True
            l_, a_, index = self.run_ops([model.loss, model.accuracy, model.opt_op], sess, train_idx, train_data, placeholders, train_flag)
            print("Update {} completed in {:.2f} sec-s".format(epoch, time.time() - start_time))
            
            if epoch != 0 and epoch % 10 == 0:
                train_flag = False
                
                l_, a_, index, pre3, tpr3, fpr3 = self.run_ops([model.loss, model.accuracy], sess, val_idx, val_data, placeholders, train_flag)
                
                train_flag = True
                l_db , a_db, index_db= self.run_ops([model.loss, model.accuracy], sess, val_idx_db, db_val, placeholders, train_flag)

                
        
        
        
        
        # test on juliet using trained model

        train_flag = False
        evaluation_start_time_part2 = time.time()
        test_loss_, test_ass_, test_index_, pre5, tpr5, fpr5 = self.run_ops([model.loss, model.accuracy], sess, test_idx, test_data, placeholders, train_flag)
        evaluation_time_part2 = time.time()-evaluation_start_time_part2


        train_flag = True
        test_db_loss_, test_db_ass_, test_db_index = self.run_ops([model.loss, model.accuracy], sess, test_idx_db, db_test, placeholders, train_flag)
        


        return [test_loss_, test_ass_, test_index_ , pre5, tpr5, fpr5], [test_db_loss_, test_db_ass_, test_db_index]

    def test_using_trained_model(self):
        # test samples using the trained model
        
        test_set = False
        
        model_path = 'saved_trained_model/cwe416/gcn_model.ckpt'
        word_dict = joblib.load('saved_trained_model/cwe416/word_dict.pkl')
        
        FEATURES_DIM = len(word_dict)




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
            model = GCN_with_dense(placeholders, input_dim=FEATURES_DIM, name="current_graph", logging=True)

        with tf.Session( graph = graph ) as sess:
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            saver.restore(sess, model_path)
            test_indexes = joblib.load('saved_trained_model/cwe416/test_indexes.pkl')
            list_binaries = joblib.load('saved_trained_model/cwe416/binary_paths.pkl')
            list_ddgs = joblib.load('saved_trained_model/cwe416/ddgs.pkl')
            list_subgraphs = joblib.load('saved_trained_model/cwe416/subgraphs.pkl')
            average = joblib.load('saved_trained_model/cwe416/average_bb_number.pkl')
            
            
            
            real_address = list_subgraphs[test_indexes[-1]].graph['debug_line_addressess']
            
            res_slices = self.slice_pdg_testset(list_ddgs[test_indexes[-1]], average, self.allowed_edges[test_indexes[-1]])
            
            adj_slices = []
            feature_slices = []
            for slic in res_slices[0]:
                adj, feat, one_hot =  self.convert_to_gcn_features_new(slic, 0)
                adj_slices.append(adj)
                feature_slices.append(sp.csr_matrix(feat))
            

            #################################################
            #list_slices = adj_testing[jdx]
            labels = [one_hot]
            
            for idx_ in range(len(adj_slices)):
                
                
                adj_slice = np.asarray([adj_slices[idx_]])
                feature_slice = np.asarray([feature_slices[idx_]])
                

                

                
                for batch in construct_batch(adj_slice, feature_slice, labels, self.batch_size):
                    
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


                    outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)

            






    