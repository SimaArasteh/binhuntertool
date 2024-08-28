import sys
import os
PROJ_PATH = os.getenv('PROJ_PATH')

""" CHECK THAT THIS IS A CLEAN REPOSITORY """
                
# EXP_NAME =
# EXP_DATE =

# experiment_setup = [
#     {"writer_name": EXP_DATE + "_" + EXP_NAME,
#     "writer_comment":"",
#     "encoder_layer_dims":[],
#     "encoder_nout":0,
#     "predictor_layer_dims":[],
#     "predictor_nfeat":0,
#     "print_every":0,
#     "report_metrics":["", ""],
#     "loss_log_str": "",
#     "acc_log_str":""}
# ]
#
# parameters = [
#     "p1",   "p2"]
# values = [
#     ("",    ""),
#     ("",    ""),
#     ("",    "")]


def get_experiment_args(args, experiment_setup, parameters, values):
    for name, value in experiment_setup.items():
        setattr(args, name, value)
    for i, set_of_values in enumerate(values):
        if args.writer_comment is None:
            args.writer_comment = str(i).zfill(2)
        else:
            args.writer_comment = str(int(args.writer_comment) + 1).zfill(2)
        for param, value in zip(parameters, set_of_values):
            setattr(args, param, value)
        # redirect stdout & stderr to a file
        experiment_outs = "{}/neural_models/autoencoders/experiment_outs".format(PROJ_PATH)
        out_file = "{path}/{date}_{name}_{suff}_out.txt".format(
            path=experiment_outs, date=args.EXP_DATE,
            name=args.EXP_NAME, suff=args.writer_comment)
        err_file = "{path}/{date}_{name}_{suff}_err.txt".format(
            path=experiment_outs, date=args.EXP_DATE,
            name=args.EXP_NAME, suff=args.writer_comment)
        if not args.debug:
            print("Redirecting out to {out_file} and err to {err_file}".format(out_file=out_file, err_file=err_file))
            sys.stdout = open(out_file, 'w', buffering=1)
            sys.stderr = open(err_file, 'w', buffering=1)
            print("Starting experiment {}_{}".format(args.EXP_DATE, args.EXP_NAME))
        yield args
