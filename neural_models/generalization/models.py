import torch
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

from torch.autograd import Variable

from node import Node
from tree import Tree
from vocab import Vocab

def Var(v):
    return Variable(v.cuda())

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    embed_size = 35
    label_size = 104
    max_epochs = 20
    amsgrad = False
    lr = 0.001
    highest_lr = 0.1
    l2 = 0.02
    
    
    
class RecursiveNN(torch.nn.Module):
    def __init__(self, vocab, config):
        super(RecursiveNN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size)
        self.W = torch.nn.Linear(2*self.config.embed_size, self.config.embed_size, bias=True)
        self.projection = torch.nn.Linear(self.config.embed_size, self.config.label_size, bias=True)
        self.activation = F.relu

    def _traverse(self, node):
        if not node:
            currentNode = Var(torch.ones(1, self.config.embed_size))
        elif node.isLeaf: 
            currentNode = self.activation(self.embedding(
                Var(torch.LongTensor([self.vocab.encode(node.word)]))))
        else: 
            l, _ = self._traverse(node.left)
            r, _ = self._traverse(node.right)
            currentNode = self.activation(
                self.W(torch.cat((l, r),1)))
        return currentNode, self.projection(currentNode)

    def forward(self, x):
        _, logits = self._traverse(x.root)
        prediction = logits.max(dim=1)[1]
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))
        return prediction, loss
    
    def set_weights_for_gen_exp(self, other):
        self.embedding.weight = other.embedding.weight
        self.W.weight = other.W.weight
        torch.nn.init.xavier_uniform(self.projection.weight)
        
        l = 0
        for child in self.children():
            if l < 2:
                for param in child.parameters():
                    param.requires_grad = False
            l += 1
        
    
    
class RecursiveNN_BN(torch.nn.Module):
    def __init__(self, vocab, config):
        super(RecursiveNN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size)
        self.W = torch.nn.Linear(2*self.config.embed_size, self.config.embed_size, bias=True)
        self.projection = torch.nn.Linear(self.config.embed_size, self.config.label_size, bias=True)
        self.activation = F.relu

    def _traverse(self, node):
        if not node:
            currentNode = Var(torch.ones(1, self.config.embed_size))
        elif node.isLeaf: 
            currentNode = self.activation(self.embedding(
                Var(torch.LongTensor([self.vocab.encode(node.word)]))))
        else: 
            l, _ = self._traverse(node.left)
            r, _ = self._traverse(node.right)
            currentNode = self.activation(
                self.W(torch.cat((l, r),1)))
        currentNode = F.normalize(currentNode)
        return currentNode, self.projection(currentNode)

    def forward(self, x):
        _, logits = self._traverse(x.root)
        prediction = logits.max(dim=1)[1]
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))
        return prediction, loss
    
    
    
class MRecursiveNN(torch.nn.Module):
    def __init__(self, vocab, config):
        super(MRecursiveNN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size**2)
        self.W = torch.nn.Linear(2*self.config.embed_size, self.config.embed_size, bias=True)
        self.projection = torch.nn.Linear(self.config.embed_size**2, self.config.label_size, bias=True)
        self.activation = F.relu

    def _traverse(self, node):
        if node.isLeaf: 
            currentNode = self.activation(self.embedding(
                Var(torch.tensor([self.vocab.encode(node.word)]))))
            currentNode = currentNode.view(self.config.embed_size, self.config.embed_size)
        else: 
            l = self._traverse(node.left).view(self.config.embed_size, self.config.embed_size)
            r = self._traverse(node.right).view(self.config.embed_size, self.config.embed_size)
            currentNode = self.activation(
                self.W(torch.cat((l, r),1)))
        return currentNode

    def forward(self, x):
        emb = self._traverse(x.root)
        logits = self.projection(emb.view(1, -1))
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))

        prediction = logits.max(dim=1)[1]
        return prediction, loss

    
    
class AdditiveRecursiveNN(torch.nn.Module):
    def __init__(self, vocab, config):
        super(AdditiveRecursiveNN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size**2)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.biases = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size)
        torch.nn.init.xavier_uniform_(self.biases.weight)
        self.projection = torch.nn.Linear(self.config.embed_size**2, self.config.label_size, bias=True)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        self.activation = F.relu

    def _traverse(self, node):
        vocab_idx = Var(torch.LongTensor([self.vocab.encode(node.word)]))
        currentNode = self.embedding(vocab_idx)
        currentNode = currentNode.view(self.config.embed_size, self.config.embed_size)
        if node.isLeaf:
            return self.activation(currentNode)
        currentBias = self.biases(vocab_idx)
        
        l = self._traverse(node.left).view(self.config.embed_size, self.config.embed_size)
        r = self._traverse(node.right).view(self.config.embed_size, self.config.embed_size)

        return self.activation(currentNode.mm(torch.add(l, r)) + currentBias)

    def forward(self, x):
        emb = self._traverse(x.root)
        logits = self.projection(emb.view(1, -1))
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))
        prediction = logits.max(dim=1)[1]
        return prediction, loss

    
    
class ResidualRecursiveNN(torch.nn.Module):
    def __init__(self, vocab, config):
        super(ResidualRecursiveNN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size)
        self.W = torch.nn.Linear(3*self.config.embed_size, self.config.embed_size, bias=True)
        self.projection = torch.nn.Linear(self.config.embed_size, self.config.label_size, bias=True)
        self.activation = F.relu
        
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def _traverse(self, node, depth=0):
        if not node:
            return Var(torch.ones(1, self.config.embed_size)), 0
        
        currentNode = self.embedding(
            Var(torch.LongTensor([self.vocab.encode(node.word)])))

        if node.isLeaf:
            return self.activation(currentNode), 0
        
        l, old_1 = self._traverse(node.left, depth + 1)
        r, old_2 = self._traverse(node.right, depth + 1)
        currentNode = self.W(torch.cat((l, r, currentNode),1))
        res = None
        if depth % 10 == 0:
            currentNode += old_1 + old_2
            res = 0
        elif (depth - 1) % 10 == 0:
            res = l+r
        else:
            res = old_1 + old_2
        return self.activation(currentNode), res
        
        
    def forward(self, x):
        emb, _ = self._traverse(x.root)
        logits = self.projection(emb)
        prediction = logits.max(dim=1)[1]
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))
        return prediction, loss
    
    
class ResidualRecursiveNN_w_N(torch.nn.Module):
    def __init__(self, vocab, config):
        super(ResidualRecursiveNN_w_N, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size)
        self.W = torch.nn.Linear(3*self.config.embed_size, self.config.embed_size, bias=True)
        self.projection = torch.nn.Linear(self.config.embed_size, self.config.label_size, bias=True)
        self.activation = F.relu
        
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def _traverse(self, node, depth=0):
        if not node:
            return Var(torch.ones(1, self.config.embed_size)), 0
        
        currentNode = self.embedding(
            Var(torch.LongTensor([self.vocab.encode(node.word)])))

        if node.isLeaf:
            currentNode = F.normalize(currentNode)
            return self.activation(currentNode), 0
        
        l, old_1 = self._traverse(node.left, depth + 1)
        r, old_2 = self._traverse(node.right, depth + 1)
        currentNode = self.W(torch.cat((l, r, currentNode),1))
        res = None
        if depth % 2 == 0:
            currentNode += old_1 + old_2
            res = 0
        elif (depth - 1) % 2 == 0:
            res = l+r
        else:
            res = old_1 + old_2
        currentNode = F.normalize(currentNode)
        return self.activation(currentNode), res
        
        
    def forward(self, x):
        emb, _ = self._traverse(x.root)
        logits = self.projection(emb)
        prediction = logits.max(dim=1)[1]
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))
        return prediction, loss
    
    
    
class MultiplicativeRecursiveNN(torch.nn.Module):
    def __init__(self, vocab, config):
        super(MultiplicativeRecursiveNN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size**2)
        self.W = torch.nn.Linear(self.config.embed_size, self.config.embed_size, bias=True)
        self.projection = torch.nn.Linear(self.config.embed_size**2, self.config.label_size, bias=True)
        self.activation = F.relu
        self.layer_norm = torch.nn.LayerNorm((1, self.config.embed_size, self.config.embed_size))
        
#         torch.nn.init.xavier_uniform_(self.W.weight)
#         torch.nn.init.xavier_uniform_(self.embedding.weight)
#         torch.nn.init.xavier_uniform_(self.projection.weight)

    def _traverse(self, node):
        if not node:
            return Var(torch.ones(self.config.embed_size, self.config.embed_size))
        
        currentNode = self.embedding(
            Var(torch.LongTensor([self.vocab.encode(node.word)])))
        currentNode = currentNode.view(self.config.embed_size, self.config.embed_size)
        
        if not node.isLeaf:
            l = self._traverse(node.left)
            r = self._traverse(node.right)
            currentNode = self.W(torch.mm(torch.mm(l, r), currentNode))
            
#         currentNode = F.normalize(currentNode)
        currentNode = self.layer_norm(currentNode.view(1, self.config.embed_size, self.config.embed_size))
        currentNode = currentNode.view(self.config.embed_size, self.config.embed_size)
        return self.activation(currentNode)
        
        
    def forward(self, x):
        emb = self._traverse(x.root)
        logits = self.projection(emb.view(1, -1))
        prediction = logits.max(dim=1)[1]
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))
        return prediction, loss
    
    def set_weights_for_gen_exp(self, other):
        self.embedding.weight = other.embedding.weight
        self.W.weight = other.W.weight
        torch.nn.init.xavier_uniform_(self.projection.weight)
        
        l = 0
        for child in self.children():
            if l < 2:
                for param in child.parameters():
                    param.requires_grad = False
            l += 1
    

    
# class TripleRecursiveNN_with_v3_trees(torch.nn.Module):
#     def __init__(self, vocab, config):
#         super(TripleRecursiveNN_with_v3_trees, self).__init__()
#         self.config = config
#         self.vocab = vocab
#         self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size)
#         self.W = torch.nn.Linear(3*self.config.embed_size, self.config.embed_size, bias=True)
#         self.projection = torch.nn.Linear(self.config.embed_size, self.config.label_size, bias=True)
#         self.activation = F.relu
        
#         torch.nn.init.xavier_uniform_(self.W.weight)
#         torch.nn.init.xavier_uniform_(self.embedding.weight)
#         torch.nn.init.xavier_uniform_(self.projection.weight)

#     def _traverse(self, node, depth=0):
#         if not node:
#             return Var(torch.ones(1, self.config.embed_size)), 0
#         currentNode = self.embedding(
#             Var(torch.LongTensor([self.vocab.encode(node.word)])))

#         if node.isLeaf:
#             return self.activation(currentNode), 0
        
#         l, old_1 = self._traverse(node.left, depth + 1)
#         r, old_2 = self._traverse(node.right, depth + 1)
#         currentNode = self.W(torch.cat((l, r, currentNode),1))
#         res = None
#         if depth % 10 == 0:
#             currentNode += old_1 + old_2
#             res = 0
#         elif (depth - 1) % 10 == 0:
#             res = l+r
#         else:
#             res = old_1 + old_2
#         return self.activation(currentNode), res
        
        
#     def forward(self, x):
#         emb, _ = self._traverse(x.root)
#         logits = self.projection(emb)
#         prediction = logits.max(dim=1)[1]
#         loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label-1])))
#         return prediction, loss


class EmptyWordRecursiveNN(torch.nn.Module):
    def __init__(self, vocab, config):
        super(EmptyWordRecursiveNN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(int(self.vocab.vocab_size), self.config.embed_size**2)
        self.W = torch.nn.Linear(self.config.embed_size, self.config.embed_size, bias=True)
        self.projection = torch.nn.Linear(self.config.embed_size**2, self.config.label_size, bias=True)
        self.activation = F.relu
        self.layer_norm = torch.nn.LayerNorm((1, self.config.embed_size, self.config.embed_size))
        
        
    def _traverse(self, node):
        if not node:
            return Var(torch.ones(self.config.embed_size, self.config.embed_size))
        
        currentNode = self.embedding(
            Var(torch.LongTensor([self.vocab.encode('UNK')])))
        currentNode = currentNode.view(self.config.embed_size, self.config.embed_size)
        
        if not node.isLeaf:
            l = self._traverse(node.left)
            r = self._traverse(node.right)
            currentNode = self.W(torch.mm(torch.mm(l, r), currentNode))
            
        currentNode = self.layer_norm(currentNode.view(1, self.config.embed_size, self.config.embed_size))
        currentNode = currentNode.view(self.config.embed_size, self.config.embed_size)
        return self.activation(currentNode)
        
        
    def forward(self, x):
        emb = self._traverse(x.root)
        logits = self.projection(emb.view(1, -1))
        prediction = logits.max(dim=1)[1]
        loss = F.cross_entropy(input=logits, target=Var(torch.tensor([x.label])))
        return prediction, loss
    
    def set_weights_for_gen_exp(self, other):
        self.embedding.weight = other.embedding.weight
        self.W.weight = other.W.weight
        torch.nn.init.xavier_uniform_(self.projection.weight)
        
        l = 0
        for child in self.children():
            if l < 2:
                for param in child.parameters():
                    param.requires_grad = False
            l += 1
    
