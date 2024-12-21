import numpy as np
import torch
import itertools
from torch.utils.data import TensorDataset, DataLoader
from sampler import LeadSum
from english_tokenizer import EnglishTokenizer
from model import SiameseComparator

class DocScore:
    def __init__(self, checkpointname, vocab, summary_length, sentence_extractor=LeadSum, remove_punctuation=False,
                 remove_stopwords=False, remove_missing=False, sentence_separation=False, tokenizer=EnglishTokenizer,
                 device="cpu", batch_size=32):

        self.device = device
        self.batch_size = batch_size
        checkpoint = torch.load(checkpointname)
        self.model = SiameseComparator(
            embeddings=checkpoint['vocab_len'],
            emb_dim=checkpoint['emb_dim'],
            nconv=checkpoint['nconv'],
            docemb_dim=checkpoint['docemb_dim']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.sextrac = sentence_extractor(summary_length)
        self.tokenizer = tokenizer(
            remove_stopwords=remove_stopwords,
            remove_punctuation=remove_punctuation,
            remove_missing=remove_missing,
            sentence_separation=sentence_separation
        )
        self.tokenizer.load_vocab(vocab)

    def score(self, docs):
        if len(docs) < 2:
            if len(docs) == 1:
                return np.ones(1)
            else:
                return np.empty(0, dtype=float)

        tokens = self._tokenize(docs)
        pairs = self._genpairs(len(docs))
        docs_dataloader = self._get_dataloader(tokens, pairs)

        pred = None
        for batch in docs_dataloader:
            x1_batch, x2_batch = tuple(t for t in batch)
            y_pred = self.model(x1_batch, x2_batch)
            y_pred = y_pred.to("cpu").detach().numpy().reshape(-1)
            if pred is None:
                pred = y_pred
            else:
                pred = np.concatenate((pred, y_pred))

        wins = np.zeros(len(docs))
        for (d1, d2), s in zip(pairs, pred):
            if s > 0.5:
                wins[d1] += 1
            else:
                wins[d2] += 1

        smin = np.min(wins)
        smax = np.max(wins)
        diffrange = smax-smin

        if diffrange > 0:
            return (wins-smin)/diffrange

        return np.ones(len(docs))


    def _tokenize(self, docs):
        preproc = [self.sextrac.extract(doc) for doc in docs]
        preproc = [[self.tokenizer.toid(sentence) for sentence in doc] for doc in preproc]
        preproc = [self.tokenizer.merge(doc) for doc in preproc]
        preproc = [self.tokenizer.pad_sequence(doc, self.sextrac.length) for doc in preproc]
        return preproc

    @staticmethod
    def _genpairs(ndocs):
        """
        Generates pairs according to a comparison strategy.
        :return:
        """
        pairs = itertools.combinations(range(ndocs), 2)
        return list(pairs)

    def _get_dataloader(self, docs, pairs):
        x1 = np.array([docs[d1] for d1, d2 in pairs])
        x2 = np.array([docs[d2] for d1, d2 in pairs])
        tensor_x1 = torch.tensor(x1).to(self.device)
        tensor_x2 = torch.tensor(x2).to(self.device)

        in_data = TensorDataset(tensor_x1, tensor_x2)
        return DataLoader(in_data, batch_size=self.batch_size)
