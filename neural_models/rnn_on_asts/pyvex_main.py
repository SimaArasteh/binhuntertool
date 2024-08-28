import argparse
import pickle as pkl
import sys
import torch

from node import Node
from train import grid_search_models, train, run_epoch
from tree import Tree
from utils import splitData
from models import RecursiveNN, Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with pyvex out.')
    parser.add_argument('--cuda', type=int, help='cuda core to run experiments on')
    parser.add_argument('--params_file', type=str, help='file containing parameters to grid search over')
    parser.add_argument('--out_file', type=str)
    
    args = parser.parse_args()
    
    params = {}
    with open(args.params_file, 'r') as f:
        for l in f.readlines():
            line_parts = l.split('=')
            assert len(line_parts) == 2, 'Unexpected line format in parameter file {}: {}'.format(
            args.params_file, l)
            params[line_parts[0].strip()] = eval(line_parts[1])
            
    # Print to file
    sys.stdout = open(args.out_file,'wt')

    # Load data
    with open('../pyvex_trees.pkl', 'rb') as f:
        trees = pkl.load(f)

    # Set cuda device
    assert args.cuda >= 0 and args.cuda < 4, 'Incorrect value for cuda: {}'.format(args.cuda)
    torch.cuda.set_device(args.cuda)
    print(torch.cuda.current_device())    

    # Do grid search
    grid_search_models(params, trees, 'pyvex_pt2_')
