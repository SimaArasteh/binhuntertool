import itertools
import numpy as np
import os
import random
import sys


def remove_comments(line, inside_comment_block):
    if '//' in line:
        ind = line.find('//')
        line = line[:ind]
    
    start_ind = line.find('/*') 
    end_ind = line.find('*/')
    
    if inside_comment_block:
        if end_ind != -1:
            line = line[end_ind + 2:]
            inside_comment_block = False
        else:
            line = ''
    
    if start_ind != -1:
        if end_ind != -1:
            line = line[:start_ind] + line[end_ind + 2:]
        else:
            line = line[:start_ind]
            inside_comment_block = True
    return line, inside_comment_block


def parse_sc_line(line, inside_comment_block=False):
    line, inside_comment_block = remove_comments(line, inside_comment_block)
    if not line:
        return '\n', inside_comment_block
    
    return line, inside_comment_block


def clean_file(filename):
    with open(filename, 'r',  errors='ignore') as f:
        lines = ''
        comment = False
        for l in f.readlines():
            l, comment = parse_sc_line(l, comment)
            lines += '\n' + l
        return lines

# def loadData(train_r=8, val_r=1, test_r=1):
#     directories = os.walk('tbcnn_data/')
#     next(directories)
#     trees = []
#     parser = c_parser.CParser()
#     for directory in directories:
#         for file in directory[2]:
#             filename = directory[0] + '/' + file
#             text = clean_file(filename)
#             try:
#                 ast = parser.parse(text, filename)
#                 all_trees = [Tree(e) for e in ast.ext]
#                 tree = combineTrees(all_trees)
#                 tree.label = int(directory[0].split('/')[1])
#                 trees.append(tree)
#             except Exception as err:
#                 print ('Unable to parse file {}'.format(filename))
#                 print (err)
#     return trees


def splitData(data, train_r=3, val_r=1, test_r=1):
    np.random.seed(42)
    data = np.random.permutation(data)
    total_parts = train_r + val_r + test_r
    one_part = len(data) // total_parts
    train_idx = one_part * train_r
    train = data[:train_idx]
    
    val_idx = train_idx + one_part * val_r
    val = data[train_idx:val_idx]
    
    test = data[val_idx:]
    return train, val, test

def getLabels(data):
    import numpy as np
    
    labels = []
    for d in data:
        labels.append(d.label)
    return labels
    assert len(np.unique(labels)) == 104
    
def generalizationExperimentSplitData(data, 
                                      common=51,
                                      common_train_r=0.8, 
                                      common_test_r=0.2,
                                      gen=-1, 
                                      gen_train_count=10,
                                      gen_val_r=0.1
                                     ):
    data = np.asarray(data)
    labels = np.asarray(getLabels(data))
    print("All labels: ", np.unique(labels))
    common_labels = np.arange(common)
    print ("Common labels: ", common_labels)
    if gen > 0:
        gen_labels = np.arange(common, common + gen)
    else:
        gen_labels = np.arange(common, max(np.unique(labels)) + 1)

    common_data = [0 if l in common_labels else 1 for l in labels]
    assert np.all(labels[np.where(not common_data)] < common)
    common_data = ma.masked_array(data, mask=common_data)
    common_data = ma.compressed(common_data)
    
    print("Unique labels of common data: ", np.unique(getLabels(common_data)))
    print("Len of common data: ", len(common_data.data))
    common_train, common_test = train_test_split(common_data, 
                                                 train_size=common_train_r, 
                                                 test_size=common_test_r, 
                                                 random_state=12)
    
    gen_data = [0 if l in gen_labels else 1 for l in labels]
    assert np.all(labels[np.where(not gen_data)] >= common)
    if gen > 0:
        assert np.all(labels[np.where(not gen_data)] < gen + common)
    gen_data = ma.masked_array(data, mask=gen_data)
    gen_data = ma.compressed(gen_data)
    print("Len of gen data: ", len(gen_data))
    
    gen_train = []
    gen_val = []
    gen_test = []
    for l in gen_labels:
        gen_data_for_one_label = []
        for d in gen_data:
            if d.label == l:
                gen_data_for_one_label.append(d)
#         print ("Len of gen_data for label {}: {}".format(l, 
#                                                          len(gen_data_for_one_label)))
                
        gen_train_l, gen_val_and_test_l = train_test_split(gen_data_for_one_label, 
                                                           train_size=gen_train_count, 
                                                           random_state=13)
        
        gen_val_l, gen_test_l = train_test_split(gen_val_and_test_l, 
                                                 train_size=gen_val_r, 
                                                 random_state=14)
        gen_train.extend(gen_train_l)
        gen_val.extend(gen_val_l)
        gen_test.extend(gen_test_l)
    print ("Size of common train: ", len(common_train))
    print ("Size of common test: ", len(common_test))
    print ("Size of gen train: ", len(gen_train))
    print ("Size of gen val: ", len(gen_val))
    print ("Size of gen test: ", len(gen_test))
    return common_train, common_test, np.asarray(gen_train), np.asarray(gen_val), np.asarray(gen_test)