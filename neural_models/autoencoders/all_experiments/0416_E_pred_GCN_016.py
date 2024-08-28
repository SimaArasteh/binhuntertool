from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "E_pred_GCN_016"
args.EXP_DATE = "0415"

# different_setups
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": None,
                    "encoder_layer_dims": None,
                    "encoder_nout": 32,
                    "predictor_nfeat": -1,
                    "predictor_layer_dims": [32],
                    "predictor_nout": 2,
                    "batch_size": 16,
                    "lr": 1e-3,
                    "epochs_save": 1,
                    "epochs_test": 1,
                    "max_epochs": 30,
                    "report_metrics": ["loss", "acc"],
                    "sparse": True}

parameters = ["encoder_layer_dims"]
values = [([32, 32, 32, 32, 32], ), ([32, 32, 32, 32, 32, 32, 32, 32], )]

for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
