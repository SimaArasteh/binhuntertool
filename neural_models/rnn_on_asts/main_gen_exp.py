import argparse
import pickle as pkl
import sys
import torch

from node import Node
from train_gen_exp import grid_search_models_gen_exp
from tree import Tree
from utils import splitData
from models import RecursiveNN, Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with pyvex out.')
    parser.add_argument('--cuda', type=int, default=0, help='cuda core to run experiments on')
    parser.add_argument('--params_file', type=str, help='file containing parameters to grid search over', default=None)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--amsgrad', type=bool, default=False)
    parser.add_argument('--emb_size', type=int, default=100)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--sc', type=bool, default=False)
    parser.add_argument('--trimmed', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='')
    
    args = parser.parse_args()
    
    params = {}
    if args.params_file is not None:
        with open(args.params_file, 'r') as f:
            for l in f.readlines():
                line_parts = l.split('=')
                assert len(line_parts) == 2, 'Unexpected line format in parameter file {}: {}'.format(
                args.params_file, l)
                params[line_parts[0].strip()] = eval(line_parts[1])
    else:
        params['embed_sizes'] = [args.emb_size]
        params['lr'] = [args.lr]
        params['l2'] = [args.l2]
        params['amsgrad'] = [args.amsgrad]
        
    # Print to file
    if args.out_file is not None:
        sys.stdout = open(args.out_file,'wt')

    # Load data
    with open(args.in_file, 'rb') as f:
        trees = pkl.load(f)
    print ('Data loaded from {}'.format(args.in_file))

    # Set cuda device
    torch.cuda.set_device(args.cuda)
    print('Using gpu #{}'.format(torch.cuda.current_device()))
    sys.stdout.flush()

    print ('Trimmed?', args.trimmed)
    # Do grid search
    grid_search_models_gen_exp(params=params, 
                               trees=trees, 
                               model_prefix=args.prefix, 
                               model_weights=args.model, 
                               sc=args.sc, 
                               trimmed=args.trimmed,
                               model_name=args.model_name)
