from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "E_pred_GCN_004"
args.EXP_DATE = "0409"

# different_setups
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": None,
                    "encoder_layer_dims": [32, 32, 32, 32, 32],
                    "encoder_nout": 32,
                    "num_edges": None}

parameters = ["num_edges"]
values = [(100, ), (500, ), (1000, ), (3000)]

for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
