from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "E_pred_MLP_006"
args.EXP_DATE = "0417"

# different_setups
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": "01",

                    "batches_log": 100,

                    "encoder": "no_encoder",
                    "epochs_save": 1,
                    "epochs_test": 1,
                    "epochs_test_start": 1,
                    "max_epochs": 5,
                    "predictor_layer_dims": [5000, 1000, 500, 100],
                    "undirected_graphs": True,

                    "batch_size": None,
                    "num_edges": None}

parameters = ["num_edges", "batch_size"]
values = [
    #(100, 8), (500, 8),
    (1000, 4), (3000, 4)]


for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
