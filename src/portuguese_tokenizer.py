import nltk
from nltk.corpus import stopwords
import string
import spacy

from tokenizer import Tokenizer

class PortugueseTokenizer(Tokenizer):
    def __init__(self, pad=0, unk=1, sep=2, pad_post=False, truncate_post=True, vocab_max_size=2000,
                 remove_missing=True, remove_punctuation=True, remove_stopwords=True, sentence_separation=True):
        super().__init__(pad, unk, sep, pad_post, truncate_post, vocab_max_size, remove_missing, remove_punctuation,
                         remove_stopwords, sentence_separation)
        self.stemmer = nltk.stem.RSLPStemmer()
        self.nlp = spacy.load('pt_core_news_lg', disable = ['parser','ner'])

    def _normalize_tokens(self, sentence):
        sentence = sentence.lower()
        doc = self.nlp(sentence)
        tokens = [tk.lemma_ for tk in doc]

        if self.remove_punctuation:
            tokens = [w for w in tokens if w not in string.punctuation + r'“”—–’']

        if self.remove_stopwords:
            stopw = stopwords.words('portuguese')
            tokens = [w for w in tokens if w not in stopw]

        return tokens

#pt_core_news_lg