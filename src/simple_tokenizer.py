import string
from tokenizer import Tokenizer

class SimpleTokenizer(Tokenizer):
    def __init__(self, pad=0, unk=1, sep=2, pad_post=False, truncate_post=True, vocab_max_size=2000,
                 remove_missing=True, remove_punctuation=True, remove_stopwords=True, sentence_separation=True,
                 stopwords=set()):
        super().__init__(pad, unk, sep, pad_post, truncate_post, vocab_max_size, remove_missing, remove_punctuation,
                         remove_stopwords, sentence_separation)
        self.stopwords = stopwords

    def _normalize_tokens(self, sentence):
        tokens = sentence.split()

        if self.remove_punctuation:
            tokens = [w for w in tokens if w not in string.punctuation + r'“”—–’']

        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords]

        return tokens
