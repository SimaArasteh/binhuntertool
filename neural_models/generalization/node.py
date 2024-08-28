class Node:
    def __init__(self, word=None, parent=None):
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
