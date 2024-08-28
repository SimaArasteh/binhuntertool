from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "S_pred_GCN_004"
args.EXP_DATE = "0427"

# different_setups
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": None,

                    "all_edges": False,
                    "batches_log": 10,
                    "batch_size": 1,
                    "encoder_nout": 32,
                    "epochs_save": 1,
                    "epochs_test": 1,
                    "epochs_test_start": 1,
                    "sample_subgraphs": True,
                    "debug": True,
                    "use_dummy_writer": True,
                    "use_roc_auc_writer": True,

                    "encoder_layer_dims": None,
                    "mask_size": None}

parameters = ["encoder_layer_dims", "mask_size"]
values = [([32, 32, 32, 32, 32], 15), ([32, 32, 32, 32, 32, 32, 32, 32], 15)]

for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
