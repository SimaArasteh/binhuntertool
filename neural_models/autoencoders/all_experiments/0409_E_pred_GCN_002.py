import sys
import os
#
# BINREC_PATH = os.getenv('BINREC_PATH')
# sys.path.append('{BINREC_PATH}/neural_models/autoencoders')
# sys.path.append('{BINREC_PATH}/neural_models/autoencoders/models')

from ..args import Args
from ..experiment_template import get_experiment_args
from ..edge_prediction_experiment import main

args = Args()

### CHECK THAT THIS IS A CLEAN REPOSITORY ###

args.EXP_NAME = "E_pred_GCN_002"
args.EXP_DATE = "0409"

# different_setups
experiment_setup = {"writer_name": args.EXP_DATE + "_" + args.EXP_NAME,
                    "writer_comment": None,
                    "encoder_layer_dims": None,
                    "encoder_nout": 64,
                    "predictor_nfeat": 128,
                    "predictor_layer_dims": [32],
                    "predictor_nout": 2,
                    "print_every": 100,
                    "report_metrics": ["loss", "acc"]}

parameters = ["encoder_layer_dims"]
values = [([256, 256, 128],),
          ([256,  256, 256, 128],)]

for args in get_experiment_args(args, experiment_setup, parameters, values):
    main(args)
