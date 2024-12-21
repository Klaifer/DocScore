import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tokenizer import Tokenizer

class EnglishTokenizer(Tokenizer):
    def _normalize_tokens(self, sentence):
        sentence = sentence.lower()
        tokens = word_tokenize(sentence)
        tagged = [(w, self._treebank2wordnet_pos(t)) for w, t in nltk.pos_tag(tokens)]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, pos=pos) for word, pos in tagged]

        if self.remove_punctuation:
            tokens = [w for w in tokens if w not in string.punctuation + r'“”—–’']

        if self.remove_stopwords:
            stopw = stopwords.words('english')
            tokens = [w for w in tokens if w not in stopw]

        return tokens

    @staticmethod
    def _treebank2wordnet_pos(tag, default=nltk.corpus.wordnet.NOUN):
        if tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return default