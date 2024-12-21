import os
import json
import numpy as np
import argparse
import h5py
from tqdm import tqdm

from scorer import DocScore
from sum_utils import getlogger, groupbydoc, truncate, RougeEval
from portuguese_tokenizer import PortugueseTokenizer
from english_tokenizer import EnglishTokenizer

def model_eval(datafile, summary_len, docscore, remove=True):
    rouge_original = RougeEval(rouge_args="-c 95 -r 1000 -n 2 -a", remove_temp=remove)  # original perl, for uniformity with other results in the literature

    with h5py.File(datafile, "r") as fin:
        for raw in tqdm(fin['dataset']):
            sample = json.loads(raw)

            # input docs
            docs = groupbydoc(sample['article'], sample['article_doc'])
            docs = list(docs.values())

            # Reference list
            references = groupbydoc(sample['abstract'], sample['abstract_doc'])
            references = ["\n".join(s) for s in references.values()]

            scores = docscore.score(docs)
            order = np.argsort(-scores)

            candidate = [s for d in order for s in docs[d]]
            candidate = truncate(candidate, summary_len)

            rouge_original.append("\n".join(candidate), references)

    return rouge_original.eval()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data', type=str, help='input data to summarize')
    parser.add_argument('model', type=str, help='trained model file')
    parser.add_argument('summary_len', type=int, help='number of extracted tokens')
    parser.add_argument('vocab', type=str, help='vocabulary file name')
    parser.add_argument('--remove_punctuation', action='store_true')
    parser.add_argument('--remove_stopwords', action='store_true')
    parser.add_argument('--remove_missing', action='store_true')
    parser.add_argument('--separate_sentences', action='store_true')
    parser.add_argument('--logfile', type=str, default="docscore_lead.log", help='Output log file')
    parser.add_argument('--lang', choices=['en', 'pt'], default='en')
    parser.add_argument('--preserve_summaries', action='store_true')

    args = parser.parse_args()

    logger = getlogger(args.logfile)
    logger.info("Filename: " + os.path.basename(__file__))
    logger.info(args)

    if args.lang == "pt":
        LanguageTokenizer = PortugueseTokenizer
    else:
        LanguageTokenizer = EnglishTokenizer

    docscore = DocScore(args.model, args.vocab, args.summary_len, remove_punctuation=args.remove_punctuation,
                        remove_stopwords=args.remove_stopwords, remove_missing=args.remove_stopwords,
                        sentence_separation=args.separate_sentences, tokenizer=LanguageTokenizer)

    metrics = model_eval(args.data, args.summary_len, docscore, remove=not args.preserve_summaries)
    logger.info(json.dumps(metrics[0], indent=2))
    logger.info(metrics[1])
