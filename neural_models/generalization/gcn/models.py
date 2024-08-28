from . import layers
from . import metrics

import tensorflow as tf 
import sys
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += 5e-4 * tf.nn.l2_loss(var)

        # Cross entropy error
#         self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
#                                                   self.placeholders['labels_mask'])

        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, 
                                                                            labels=self.placeholders['labels']))
    def _accuracy(self):
#         self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
#                                         self.placeholders['labels_mask'])
        self.accuracy = metrics.accuracy(self.outputs, self.placeholders['labels'])

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                 output_dim=16,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(layers.Dense(input_dim=16,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.pooling_matrix = placeholders['pooling']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

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
        self.outputs = self.activations[-1]
        self.outputs = tf.sparse.sparse_dense_matmul(self.pooling_matrix, self.outputs)
        
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        
    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += 5e-4 * tf.nn.l2_loss(var)

        self.print1_op = tf.print("outputs: ", tf.shape(self.outputs), output_stream=sys.stderr)
        self.print2_op = tf.print("\nlabels: ", tf.shape(self.placeholders['labels']), output_stream=sys.stderr)
        # Cross entropy error
        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.placeholders['labels'], 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy_all)
        
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
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self, pooling_matrix):
        node_preds = tf.nn.softmax(self.outputs)
        return graph_preds
    
    

class GCNA(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCNA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.pooling_matrix = placeholders['pooling']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        
        self.attention_layers = []
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
        self.outputs = self.activations[-1]
        
        attn_activations = [self.inputs]
        for layer in self.attention_layers:
            hidden = layer(attn_activations[-1])
            attn_activations.append(hidden)
        attention_weight = tf.reshape(attn_activations[-1], [-1])
        print ('attention weight ', attention_weight.shape, type(attention_weight))
        print ('pooling matrix ', type(self.pooling_matrix))
        weighted_pooling = self.pooling_matrix.__mul__(attention_weight)
        print ('weighted pooling ', weighted_pooling.shape, type(attention_weight))
        self.outputs = tf.sparse.sparse_dense_matmul(weighted_pooling, self.outputs)
        print ('outputs ', self.outputs.shape, type(attention_weight))
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        
    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += 5e-4 * tf.nn.l2_loss(var)

        self.print1_op = tf.print("outputs: ", tf.shape(self.outputs), output_stream=sys.stderr)
        self.print2_op = tf.print("\nlabels: ", tf.shape(self.placeholders['labels']), output_stream=sys.stderr)
        # Cross entropy error
        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.placeholders['labels'], 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy_all)
        
    def _build(self):
        self.layers.append(layers.GraphConvolution(input_dim=self.input_dim,
                                            output_dim=16,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(layers.GraphConvolution(input_dim=16,
                                                    output_dim=self.output_dim,
                                                    placeholders=self.placeholders,
                                                    act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging))
        
        self.attention_layers.append(layers.Dense(input_dim=self.input_dim,
                                                 output_dim=1,
                                                 sparse_inputs=True,
                                                 placeholders=self.placeholders,
                                                 act=tf.nn.tanh,
                                                 dropout=True,
                                                 logging=self.logging))

    def predict(self, pooling_matrix):
        node_preds = tf.nn.softmax(self.outputs)
        return graph_preds
    
