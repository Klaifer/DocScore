# Document evaluation occurs on a subset of their sentences.
# The selection of sentences to be evaluated can be done with a single-document extractive summarization technique.
from sum_utils import truncate

class SentenceExtractor():
    def __init__(self, length):
        self.length = length

    def extract(self, sentences):
        raise NotImplementedError()

class LeadSum(SentenceExtractor):
    """
    Single document summarizer applied in document evaluation
    """
    def extract(self, sentences):
        truncated = truncate(sentences, self.length)
        return truncated