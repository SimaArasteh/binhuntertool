import os

### program configuration
class Args():
    def __init__(self):
        ### if clean tensorboard
        self.clean_tensorboard = False
        ### Which CUDA GPU device is used for training
        self.cuda = int(os.getenv("CUDA"))
        assert self.cuda >= 0 and self.cuda <4
        print("Training on GPU: ", self.cuda)

        ### What model to use for graph representation
        self.encoder = 'gcn'
        
        ### What model to use for prediction
        self.predictor = 'fc'
        
        ### Which dataset is used in the experiment
        # old - the tokens are truncated by frequency
        # two_digit - the tokens of constants are truncated to least significant (last) 2 digits
        # const - the tokens of constants are all replaced with a placeholder token CONST
        self.data = 'old'
        #  self.data = 'two_digit'
        # self.data = 'const'

        ### Whether to test on test or validation set
        self.use_test_set = False
        #  Execute unit tests in the code
        self.run_tests = False

        ### Experiment context
        # Whether to use all predictions for all nodes to calculate the loss
        self.calculate_loss = 'on_all'

        ### network config
        ## GCN
        self.encoder_nfeat = None # depends on the data type and fold
        self.encoder_layer_dims = [512, 256, 256]
        self.encoder_nout = 128
        self.encoder_softmax = True

        ## FC
        self.predictor_nfeat = 256
        self.predictor_layer_dims = [32]
        self.predictor_nout = 2
        
        ### training config
        self.max_epochs = 30
        self.batch_size = 16
        self.epochs_test_start = 5
        self.epochs_test = 2
        self.batches_log = 500
        self.epochs_log = 1
        self.epochs_save = 5
        self.lr = 1e-3
        self.report_metrics = ["loss", "acc"]
        self.shuffle_after_epoch = False
        # self.milestones = [400, 1000]
        # self.lr_rate = 0.99

        ### debugging config
        self.use_entire_training_set = True
        self.num_training_examples = None
        self.use_entire_testing_set = True
        self.num_testing_examples = None
        self.debug = False

        self.logs_dir = "neural_models/autoencoders/runs/"
        self.model_ckp_dir = "neural_models/autoencoders/model_checkpoints/"
        self.allow_load_pretrained_model = False

        ### Experiment parameters
        # Edge prediction
        self.num_edges = 1000
        self.edge_masking_value = 0
        self.undirected_graphs = False
        self.encoder = None
        self.sample_subgraphs = False
        self.mask_size = 5
        self.all_edges = False
        self.use_dummy_writer = False
        self.use_roc_auc_writer = False

        self.add_self_loops = False

