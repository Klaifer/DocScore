import os
import nltk
import logging
from pyrouge import Rouge155
import random
import shutil
import string

def unusedname(filename):
    if os.path.isfile(filename):
        parts = list(os.path.splitext(filename))
        base = parts[0]
        counter = 2

        parts[0] = base+str(counter)
        filename = "".join(parts)
        while os.path.isfile(filename):
            counter+=1
            parts[0] = base + str(counter)
            filename = "".join(parts)

    return filename


def getlogger(filename, overwrite=False):
    if not overwrite:
        filename = unusedname(filename)

    folder = os.path.dirname(filename)
    try:
        os.makedirs(folder)
    except (FileExistsError, FileNotFoundError):
        pass

    # Stdout config
    format = '%(asctime)s %(levelname)s %(name)s %(message)s'
    datefmt = '%d/%m/%Y %H:%M:%S'
    logging.basicConfig(format=format, datefmt=datefmt, level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger()

    # File config
    formatter = logging.Formatter(format, datefmt=datefmt)
    logfile = logging.FileHandler(filename)
    logfile.setFormatter(formatter)
    logfile.setLevel(logging.INFO)
    logger.addHandler(logfile)

    return logger


def truncate(sentences, ntokens):
    summary = []
    stokens = 0

    for i, s in enumerate(sentences):
        tokens = nltk.word_tokenize(s)
        ctokens = len(tokens)
        if stokens + ctokens < ntokens:
            summary.append(s)
            stokens += ctokens
        else:
            summary.append(" ".join(tokens[:ntokens - stokens]))
            break
    return summary

def groupbydoc(sentences, docid):
    docs = {}
    for did, sent in zip(docid, sentences):
        docs[did] = docs.get(did, []) + [sent]

    return docs

class RougeEval:
    def __init__(self, rouge_args=None, remove_temp=True):
        self.rouge_args = rouge_args
        self.remove_temp = remove_temp

        self.rouge = Rouge155()
        self.ndocs = 0
        self.rouge.log.setLevel(logging.WARNING)

        self._prepare_dirs()


    def __del__(self):
        if self.remove_temp:
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logging.warning(e)

        try:
            shutil.rmtree(self.rouge._config_dir)
        except Exception as e:
            logging.warning(e)

        try:
            shutil.rmtree(os.path.dirname(self.rouge._system_dir))
        except Exception as e:
            logging.warning(e)

    def append(self, candidate, references):
        """
        Includes a file to Rouge Evaluation.
        It will prepare documentos to eval method

        :param candidate: Generated summary
        :param references: List of reference sumaries
        """
        for i, ref in enumerate(references):
            reffilename = "{}.{}.txt".format(self.ndocs, i)
            with open(os.path.join(self.referencepath, reffilename), "w") as fin:
                fin.write(ref)

        summaryfilename = "{}.txt".format(self.ndocs)
        with open(os.path.join(self.summarypath, summaryfilename), "w") as fin:
            fin.write(candidate)

        self.ndocs += 1


    def eval(self):
        self.rouge.system_dir = self.summarypath
        self.rouge.model_dir = self.referencepath
        self.rouge.system_filename_pattern = '(\d+).txt'
        self.rouge.model_filename_pattern = '#ID#.\d+.txt'

        if self.rouge_args:
            self.rouge_args = '-e {} {}'.format(self.rouge.data_dir, self.rouge_args)

        output = self.rouge.convert_and_evaluate(rouge_args=self.rouge_args)
        r = self.rouge.output_to_dict(output)

        return r, output


    def _prepare_dirs(self):
        self.temp_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        self.temp_dir = os.path.join("rouge_temp", self.temp_dir)

        while os.path.exists(self.temp_dir):
            self.temp_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            self.temp_dir = os.path.join("rouge_temp", self.temp_dir)

        self.summarypath = os.path.join(self.temp_dir, 'summary')
        self.referencepath = os.path.join(self.temp_dir, 'reference')

        # directory for generated summaries
        os.makedirs(self.summarypath)

        # directory for reference summaries
        os.makedirs(self.referencepath)