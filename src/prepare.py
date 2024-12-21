import argparse
import os
import h5py
import random
import json
from rouge_score import rouge_scorer
from tqdm import tqdm

from english_tokenizer import EnglishTokenizer
from portuguese_tokenizer import PortugueseTokenizer
from sum_utils import groupbydoc
from sampler import LeadSum


class PairedDataIterator:
    """
    Receives a h5df dataset, and changes to a siamese classification problem
    """

    def __init__(self, filename, summ, text_conversor):
        self.datafile = h5py.File(filename, "r")
        self.text_conversor = text_conversor
        self.summarizer = summ

        self.ndocs = len(self.datafile['dataset'])
        self.currdoc = 0
        self.pairs = []
        self.rouge_fast = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def __iter__(self):
        self.dataset = enumerate(self.datafile['dataset'])
        return self

    def __del__(self):
        try:
            self.datafile.close()
        except AttributeError:
            pass

    def __next__(self):
        while True:
            doc1, doc2, label, score1, score2 = self._next_sample()

            doc1 = [self.text_conversor.toid(s) for s in doc1]
            doc1 = self.text_conversor.merge(doc1)
            if not doc1:  # Doc can be empty after removing missing words
                continue
            doc1 = self.text_conversor.pad_sequence(doc1, self.summarizer.length)

            doc2 = [self.text_conversor.toid(s) for s in doc2]
            doc2 = self.text_conversor.merge(doc2)
            if not doc2:  # Doc can be empty after removing missing words
                continue
            doc2 = self.text_conversor.pad_sequence(doc2, self.summarizer.length)

            break

        return doc1, doc2, label, self.currdoc, score1, score2

    def _next_sample(self):
        if not self.pairs:
            self._updatepairs()

        (idoc1, idoc2) = self.pairs.pop()
        doc1 = self.docs[idoc1]
        doc2 = self.docs[idoc2]
        label = 1 if self.scores[idoc1] > self.scores[idoc2] else 0

        return doc1, doc2, label, self.scores[idoc1], self.scores[idoc2]

    def _updatepairs(self):
        self.docs = {}
        sample = None
        # Avoid single document samples
        while len(self.docs) < 2:
            self.currdoc, raw = next(self.dataset)
            sample = json.loads(raw)
            self.docs = groupbydoc(sample['article'], sample['article_doc'])
        docids = list(self.docs.keys())

        # Reference list
        references = groupbydoc(sample['abstract'], sample['abstract_doc'])
        references = ["\n".join(s) for s in references.values()]

        # Score documents
        self.scores = {}
        for idoc, doc in self.docs.items():
            candidate = "\n".join(self.summarizer.extract(doc))
            score = self.rouge_fast.score_multi(references, candidate)
            self.scores[idoc] = score['rouge1'].fmeasure

        # Generates pairs avoiding document repetition
        random.shuffle(docids)
        self.pairs = list(zip(docids[::2], docids[1::2]))


def toh5df(data_iterator, outputfile, max_samples=-1):
    """
    Salva os dados formatados para um arquivo de saída.

    :param data_iterator:
    :param outputfile:
    :param max_samples
    :return:
    """

    # 1 - Read data
    doc1 = []
    doc2 = []
    label = []
    sampleid = []
    score1 = []
    score2 = []

    if max_samples > 0:
        niter = max_samples
    else:
        niter = data_iterator.ndocs

    with tqdm(total=niter, desc="Generating samples", dynamic_ncols=True) as pbar:
        for d1, d2, l, did, s1, s2 in data_iterator:
            doc1.append(d1)
            doc2.append(d2)
            label.append(l)
            sampleid.append(did)
            score1.append(s1)
            score2.append(s2)

            pbar.update(data_iterator.currdoc - pbar.last_print_n)

            if max_samples > 0:
                if data_iterator.currdoc >= max_samples:
                    break

    # 2 - Balances class count
    positives = sum(label)
    unbalanced = int((len(label) / 2 - positives))

    if unbalanced < 0:
        rindex = [i for i, v in enumerate(label) if v == 1]
    else:
        rindex = [i for i, v in enumerate(label) if v == 0]

    random.shuffle(rindex)
    for i in range(abs(unbalanced)):
        curr = rindex[i]
        label[curr] = 1 - label[curr]
        balde = doc1[curr]
        doc1[curr] = doc2[curr]
        doc2[curr] = balde

    # 3 - Write to file
    fout = h5py.File(outputfile, "w")
    try:
        fout.create_dataset("doc1", data=doc1)
        fout.create_dataset("doc2", data=doc2)
        fout.create_dataset("label", data=label)
        fout.create_dataset("sampleid", data=sampleid)
        fout.create_dataset("score1", data=score1)
        fout.create_dataset("score2", data=score2)
    except Exception as e:
        print(e)

    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('infile', type=str, help='input data file')
    parser.add_argument('outfile', type=str, help='data for tunining hyperparameters')
    parser.add_argument('summary_length', type=int, help='number of extracted tokens')
    parser.add_argument('vocab', type=str, help='Vocabulary file name. Generates when file not found')
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--remove_punctuation', action='store_true')
    parser.add_argument('--remove_stopwords', action='store_true')
    parser.add_argument('--remove_missing', action='store_true')
    parser.add_argument('--separate_sentences', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--lang', choices=['en', 'pt'], default='en')
    args = parser.parse_args()
    print(args)

    if args.lang == "pt":
        LanguageTokenizer = PortugueseTokenizer
    else:
        LanguageTokenizer = EnglishTokenizer

    ltokenizer = LanguageTokenizer(
        vocab_max_size=args.vocab_size,
        remove_stopwords=args.remove_stopwords,
        remove_punctuation=args.remove_punctuation,
        remove_missing=args.remove_missing,
        sentence_separation=args.separate_sentences
    )

    if os.path.isfile(args.vocab):
        print("Loading vocab")
        ltokenizer.load_vocab(args.vocab)
    else:
        ltokenizer.extract_vocab(args.infile)
        ltokenizer.save_vocab(args.vocab)

    random.seed(args.seed)
    summarizer = LeadSum(args.summary_length)
    dataIterator = PairedDataIterator(args.infile, summarizer, ltokenizer)
    toh5df(dataIterator, args.outfile, args.max_samples)
