import numpy as np
from collections import Counter
import json
import h5py
from tqdm import tqdm

class Tokenizer:
    def __init__(self, pad=0, unk=1, sep=2, pad_post=False, truncate_post=True, vocab_max_size=2000,
                 remove_missing=True, remove_punctuation=True, remove_stopwords=True, sentence_separation=True):
        self.pad = pad
        self.unk = unk
        self.sep = sep
        self.remove_missing = remove_missing
        self.pad_post = pad_post
        self.truncate_post = truncate_post
        self.vocab_max_size = vocab_max_size
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.sentence_separation = sentence_separation

    def toid(self, sentence):
        tokens = self._normalize_tokens(sentence)
        if self.remove_missing:
            tokenid = [self.w2i[w] for w in tokens if w in self.w2i]
        else:
            tokenid = [self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in tokens]

        return tokenid

    def merge(self, sequences):
        merged = sequences[0]
        for seq in sequences[1:]:
            if self.sentence_separation:
                merged = merged + [self.sep] + seq
            else:
                merged = merged + seq

        return merged

    def pad_sequence(self, sequence, maxlen, dtype="int32"):
        """
            Some code borrowed from "tensorflow"

            :param sequences
            :param dtype
        """
        x = np.full(maxlen, self.pad, dtype=dtype)
        if self.truncate_post:
            trunc = sequence[:maxlen]
        else:
            trunc = sequence[-maxlen:]

        if self.pad_post:
            x[: len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc

        return x

    def extract_vocab(self, filename):
        w2i = Counter()
        with h5py.File(filename, "r") as datafile:
            for raw in tqdm(datafile['dataset'], desc="Generating vocab", dynamic_ncols=True):
                sample = json.loads(raw)
                for sentence in sample['article']:
                    tokens = self._normalize_tokens(sentence)
                    w2i.update(tokens)

        self.w2i = {k: i for i, (k, c) in enumerate(w2i.most_common(self.vocab_max_size), 3)}
        self.w2i['<pad>'] = self.pad
        self.w2i['<unk>'] = self.unk
        self.w2i['<sep>'] = self.sep

    def save_vocab(self, fileout):
        words = self.w2i.items()
        words = sorted(words, key=lambda x: x[1])

        with open(fileout, "w", encoding="utf-8") as fout:
            for word, _ in words:
                fout.write(word + "\n")

    def load_vocab(self, filein):
        with open(filein, "r") as fin:
            words = fin.readlines()
            self.w2i = {t.replace("\n", ""): i for i, t in enumerate(words)}
            self.vocab_max_size = len(words)

    def _normalize_tokens(self, sentence):
        raise "Not implemented"