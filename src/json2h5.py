import argparse
import json
import nltk
import h5py
from tqdm import tqdm


def split_sentences(articles, sentsplit):
    articles = [sentsplit.tokenize(doc) for doc in articles]
    articleids = [[i for _ in doc] for i, doc in enumerate(articles)]
    articleids = [idoc for doc in articleids for idoc in doc]
    articles = [sent for doc in articles for sent in doc]

    return articles, articleids


def toh5(inname, outname, sentsplit):
    with open(inname, "r", encoding="utf-8") as fin:
        content = json.load(fin)

    with h5py.File(outname, "w") as fout:
        dt = h5py.special_dtype(vlen=str)
        fout.create_dataset("dataset", (len(content),), dtype=dt)
        dataset = fout["dataset"]
        for isample, sample in enumerate(tqdm(content, dynamic_ncols=True)):
            article, article_doc = split_sentences(sample['source'], sentsplit)
            abstract, abstract_doc = split_sentences(sample['reference'], sentsplit)
            dataset[isample] = json.dumps({
                'article': article,
                'article_doc': article_doc,
                'abstract': abstract,
                'abstract_doc': abstract_doc
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/MultiNews/multinews.train.json', help='Input json file')
    parser.add_argument('--output', type=str, default='../data/MultiNews/h5/multinews.train.h5df',
                        help='Output h5 file')
    parser.add_argument('--lang', choices=['en', 'pt'], default="en", help='content language')
    args = parser.parse_args()

    print(args)

    if args.lang == 'pt':
        print("Loading Portuguese punkt")
        sentencesplit = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    else:
        print("Loading English punkt")
        sentencesplit = nltk.data.load('tokenizers/punkt/english.pickle')

    toh5(args.input, args.output, sentencesplit)
