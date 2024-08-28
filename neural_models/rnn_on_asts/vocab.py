from collections import defaultdict

class Vocab():
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_to_label = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.vocab_size = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, None, count=0)
    
    def add_word(self, word, label, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            if label is not None:
                self.word_to_label[word] = [label]
            else:
                self.word_to_label[word] = None
        
        if label is not None:
            if label not in self.word_to_label[word]:
                self.word_to_label[word].append(label)
        else:
            self.word_to_label[word] = None
        self.word_freq[word] += count
    
    def add_all_children_words(self, node, label):
        if node is None:
            return
        self.add_word(node.word, label)
        if not node.isLeaf:
            self.add_all_children_words(node.left, label)
            self.add_all_children_words(node.right, label)
    
    def construct_from_trees(self, trees):
        for t in trees:
            self.add_all_children_words(t.root, t.label)
#         self.trim_vocab_by_labels(10)
        self.total_words = float(sum(self.word_freq.values()))
        self.vocab_size = len(self.word_freq)
        print ('{} total words with {} uniques'.format(
            self.total_words, self.vocab_size))
    
    def trim_vocab_by_labels(self, min_labels=5):
        new_word_to_index = {}
        new_index_to_word = {}
        new_word_freq = defaultdict()
        for w, l in self.word_to_label.items():
            if l is None or len(l) >= min_labels:
                index = len(new_word_to_index)
                new_word_to_index[w] = index
                new_index_to_word[index] = w
                new_word_freq[w] = self.word_freq[w]
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq = new_word_freq
        
#         self.total_words = float(sum(self.word_freq.values()))
#         self.vocab_size = len(self.word_freq)
#         print ('{} total words with {} uniques'.format(
#             self.total_words, self.vocab_size))
                
    
    def construct(self, words):
        for word in words:
            self.add_word(word, None)
        self.total_words = float(sum(self.word_freq.values()))
        self.vocab_size = len(self.word_freq)
        print ('{} total words with {} uniques'.format(
            self.total_words, self.vocab_size))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]
    
    def __len__(self):
        return len(self.word_freq)
    
    
