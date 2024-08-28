from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "E_pred_GCN_015"
args.EXP_DATE = "0416"

# different_setups
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": "01",
                    "encoder_nout": 32,
                    "encoder_layer_dims": [32, 32, 32, 32, 32],
                    "undirected_graphs": None,
                    "report_metrics": ["loss", "acc"]}

parameters = ["undirected_graphs"]
values = [(True, ),]

for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
