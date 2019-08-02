
class Vocabulary:
    def __init__(self, pad='<pad>', unk='<unk>', min_freq=6):
        self._idx2word = []
        self._word2idx = {}
        self._word2freq = {}
        self._len = 0
        self._pad = pad
        self._unk = unk
        self._min_freq = min_freq

    def __len__(self):
        return self._len

    def add(self, word):
        self._idx2word.append(word)
        self._word2idx[word] = self._len
        self._len += 1
        if word in self._word2freq:
            self._word2freq[word] += 1
        else:
            self._word2freq[word] = 1

    def from_dataset(self, dataset):
        for text in dataset:
            for word in text:
                self.add(word)

    def build(self):
        idx2word = [self._pad, self._unk]
        word2idx = {
            self._pad: 0,
            self._unk: 1
        }
        length = 2
        for word in self._word2freq:
            if word not in word2idx and self._word2freq[word] >= self._min_freq:
                idx2word.append(word)
                word2idx[word] = length
                length += 1
        # update
        self._idx2word = idx2word
        self._word2idx = word2idx
        self._len = length
        assert self._len == len(self._word2idx) == len(self._idx2word)

    def idx2word(self, idx):
        if idx >= self._len:
            return self._unk
        else:
            return self._idx2word[idx]

    def word2idx(self, word):
        if word not in self._word2idx:
            return self._word2idx[self._unk]
        else:
            return self._word2idx[word]
