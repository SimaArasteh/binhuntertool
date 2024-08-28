import glob
import numpy as np
import pickle as pkl

def convertLabelsForClassification(labels):
    converted_labels = []
    new_labels_mapping = {}
    for l in labels:
        if l not in new_labels_mapping:
            new_labels_mapping[l] = len(new_labels_mapping)
        converted_labels.append(new_labels_mapping[l])
    return np.asarray(converted_labels)


def read_graphs_from_file(pkl_file):
    with open(pkl_file, 'rb') as f:
        graph_pairs = pkl.load(f)
    graphs = []
    file_numbers = []
    graph_labels = []
    for graph, filename in graph_pairs:
        graphs.append(graph)
        file_numbers.append(int(filename.split('/')[-1].split('.')[0]))
        graph_labels.append(int(filename.split('/')[-2]))
    return graphs, graph_labels, file_numbers


def read_all_graphs(cc, olvl):
    parentdir = '/nas/home/shushan/data/shushan/inv_data/graphs/{}_bin_{}/*'.format(cc, olvl)
    all_files = glob.glob(parentdir)

    all_graphs = []
    all_file_numbers = []
    all_labels = []
    for f in all_files:
        graphs, graph_labels, file_numbers = read_graphs_from_file(pkl_file=f)
        all_graphs.append(graphs)
        all_file_numbers.append(file_numbers)
        all_labels.append(graph_labels)
    all_graphs = [i for sublist in all_graphs for i in sublist]
    all_file_numbers = [i for sublist in all_file_numbers for i in sublist]
    all_labels = [i - 1 for sublist in all_labels for i in sublist]
    return all_graphs, all_file_numbers, all_labels

