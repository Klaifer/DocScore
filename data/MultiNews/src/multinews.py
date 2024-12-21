import argparse
import json
from tqdm import tqdm

class MultinewsReader():
    def __init__(self, sourcefilename, targetfilename, filter_small=False):
        """

        :param sourcefilename:  Reference paragraphs for summary construction
        :param targetfilename: Reference summary
        :param embfilename: Sentence Embeddings
        :param filter_small: Remove samples with less than two sources
        """
        self.sourcefilename = sourcefilename
        self.targetfilename = targetfilename
        self.filter_small = filter_small

    def __del__(self):
        self.source.close()
        self.target.close()

    def __iter__(self):
        self.source = open(self.sourcefilename, 'r')
        self.target = open(self.targetfilename, 'r')

        return self

    def __next__(self):

        while True:
            samplesrc = self.source.readline()
            sampletgt = self.target.readline()

            if not sampletgt:
                raise StopIteration

            sampletgt = MultinewsReader.split_docs(sampletgt)
            samplesrc = MultinewsReader.split_docs(samplesrc)

            nsources = len({d for d in samplesrc})

            if not self.filter_small:
                break
            elif nsources > 2:
                break


        return {
            'source': samplesrc,
            'reference': sampletgt,
        }

    @staticmethod
    def split_docs(text):
        """
        Split text in sentences and document ids.
        Empty documents are removed, lacking document id

        :param text:
        :return:
        """
        articles = text.replace("NEWLINE_CHAR", "\n")
        articles = articles.split("story_separator_special_tag")
        articles = [s for a in articles for s in a.split("|||||")]
        articles = [s.strip() for s in articles]

        return [s for s in articles if s != ""]

def tojson(input_src, input_tgt, outputfile, filter_small=False):
    data_iterator = MultinewsReader(input_src, input_tgt, filter_small)

    dataset = []
    for i, item in tqdm(enumerate(data_iterator), dynamic_ncols=True):
        item['id'] = i
        dataset.append(item)

    with open(outputfile, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(dataset, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_src', type=str, default='../raw_version/test.raw.src', help='Input source file')
    parser.add_argument('--input_tgt', type=str, default='../raw_version/test.raw.tgt', help='Input target file')
    parser.add_argument('--output', type=str, default='../multinews.test.json', help='Output json file')
    parser.add_argument('--filter_small', action='store_true', help='Remove samples with less than two sources')
    args = parser.parse_args()

    print(args)
    tojson(args.input_src, args.input_tgt, args.output, args.filter_small)