# Oracle when evaluating incoming documents. Once the best document is selected, it summarizes with LEAD.
import os
import json
import argparse
import random

import h5py
from tqdm import tqdm

from sum_utils import getlogger, groupbydoc, truncate, RougeEval

def lead_eval(datasetfile, doclen, original_order=False, sample_scores=False, remove=True):
    if sample_scores:
        rouge_original = RougeEval(rouge_args="-d -c 95 -r 1000 -n 2 -a", remove_temp=remove)
    else:
        rouge_original = RougeEval(rouge_args="-c 95 -r 1000 -n 2 -a", remove_temp=remove)

    with h5py.File(datasetfile, "r") as data_file:
        for dt in tqdm(data_file['dataset']):
            sample = json.loads(dt)

            references = groupbydoc(sample['abstract'], sample['abstract_doc'])
            references = ["\n".join(s) for s in references.values()]

            docs = groupbydoc(sample['article'], sample['article_doc'])
            docid = list(docs.keys())
            if not original_order:
                random.shuffle(docid)

            candidate = [s for did in docid for s  in docs[did]]
            candidate = truncate(candidate, doclen)
            rouge_original.append("\n".join(candidate), references)

        return rouge_original.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='database file name')
    parser.add_argument('truncate', type=int, help='summary lenght')
    parser.add_argument('--logfile', type=str, default="../logs/lead.log", help='Output log file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed to input doc order')
    parser.add_argument('--preserve_order', action='store_true', help='Use the original document order')
    parser.add_argument('--sample_scores', action='store_true', help='add per evaluation scores')
    parser.add_argument('--preserve_summaries', action='store_true')
    args = parser.parse_args()

    logger = getlogger(args.logfile)
    logger.info("Filename: " + os.path.basename(__file__))
    logger.info(args)

    random.seed(args.seed)
    metrics = lead_eval(args.dataset, args.truncate, original_order=args.preserve_order, sample_scores=args.sample_scores, remove=not args.preserve_summaries)
    logger.info(json.dumps(metrics[0], indent=2))
    logger.info(metrics[1])