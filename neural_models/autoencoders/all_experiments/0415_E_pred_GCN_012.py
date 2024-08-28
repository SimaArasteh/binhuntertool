from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "E_pred_GCN_012"
args.EXP_DATE = "0415"


# different_setups
# the values overwritten in the course of experiment are set to None
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": '07', # can be either None, or a string with number
                    "encoder_layer_dims": [512, 128, 128],
                    "encoder_nout": 128,
                    "predictor_nfeat": -1,
                    "encoder_softmax": None,
                    "use_entire_training_set": None,
                    "num_training_examples": None,
                    "predictor_layer_dims": [32],
                    "predictor_nout": 2,
                    "print_every": 100,
                    "epochs_save": 1000,
                    "epochs_test": 10,
                    "max_epochs": 100,
                    "batches_log": 2,
                    "report_metrics": ["loss", "acc"]}

parameters = ["encoder_softmax",
              "use_entire_training_set", "num_training_examples",
              "use_entire_testing_set", "num_testing_examples",
              "lr"]
values = [(False,
           False, 10,
           False, 10,
           1e-3)]

for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
