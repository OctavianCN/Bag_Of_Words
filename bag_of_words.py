from data import *
from scipy.sparse import dok_matrix
class BagOfWords:

    def __init__(self):
        self.vocab = {}
        self.words = []
        self.list = []

    def build_vocabulary(self, train_data):
        for sentence in train_data.lista:
            for word in sentence:
                word = word.lower()
                if word not in self.vocab and len(word) > 2:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)

    def get_features(self, data):
        features = dok_matrix((len(data.lista), len(self.words)))
        for idx, sentence in enumerate(data.lista):
            for word in sentence:
                if word in self.vocab:
                    features[idx, self.vocab[word]] += 1
        return features