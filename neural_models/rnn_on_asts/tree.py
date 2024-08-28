import itertools
import numpy as np
import os
import sys
import random

from collections import defaultdict
from node import Node

class Tree:
    def __init__(self, func=None, label=None):
        if func is not None:
            self.root = self.parse(func)
        self.label = label
        
    def parse(self, obj, parent=None, idx=0):
        node = Node(getWord(obj), parent)
        children = obj.children()
        nchildren = len(children) - idx
        if nchildren > 2:
            node.left = self.parse(children[idx][1], node)
            node.right = self.parse(obj, parent, idx=idx+1)
        elif nchildren == 2:
            node.left = self.parse(children[0][1], node)
            node.right = self.parse(children[1][1], node)
        else:
            node.isLeaf = True
            if nchildren == 1:
                node.word += ' ' + getWord(children[0][1])
        return node
    
    def getWords(self):
        leaves = getLeaves(self.root)
        return [node.word for node in leaves]
    
def combineTrees(trees):
    ntrees = len(trees)
    if ntrees == 1:
        return trees[0]
    
    nodes = []
    for i in range(ntrees - 1):
        nodes.append(Node())
    
    for i in range(len(nodes)):
        nodes[i].left = trees[i].root
        if i + 1 == len(nodes):
            nodes[i].right = trees[i + 1].root
        else:    
            nodes[i].right = nodes[i + 1]
        if i > 0:
            nodes[i].parent = nodes[i - 1]

    tree = Tree()
    tree.root = nodes[0]
    return tree
    
def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)

def getWord(obj, verbose=False):
    word = obj.__class__.__name__ + ' '
#     if obj.attr_names:
#         if verbose:
#             nvlist = [(n, getattr(obj,n)) for n in obj.attr_names]
#             attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
#         else:
#             vlist = [getattr(obj, n) for n in obj.attr_names]
#             attrstr = ', '.join('%s' % v for v in vlist)
#         word += attrstr
    return word
