from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "E_pred_MLP_005"
args.EXP_DATE = "0417"

# different_setups
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": None,

                    "allow_load_pretrained_model": True,
                    "batches_log": 100,
                    "batch_size": 8,
                    "epochs_save": 1,
                    "epochs_test": 1,
                    "epochs_test_start": 1,
                    "predictor_layer_dims": [5000, 1000, 500, 100],

                    "undirected_graphs": None,
                    "encoder": None}

parameters = [
    "encoder", "undirected_graphs"]
values = [
    ("no_encoder", True)
          ]


for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
