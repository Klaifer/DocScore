# Document evaluation for summarization.

Code for Bracis: Siamese Network-Based Prioritization for Enhanced Multi-Document Summarization

## Instalation
1. Save a copy of this repository;
2. Install the packages in requirements.txt;
3. Download and unzip the [fasttext](https://fasttext.cc/docs/en/english-vectors.html);

## Other dependencies
1. When running for the first time, nltk needs some additional files.
```
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

2. For a better comparison with the literature, we are using the original ROUGE, in Perl, which is executed through a wrapper (pyrouge). Therefore, it is necessary that ROUGE is running.
As an option, it is possible to change the code to the pure Python version that we also use because it is faster (rouge-score).

## Data

We store the data in h5df format. For simplicity, we include a script to convert json files to h5df. To be applied, the file must be in the following format:
```
{
    "id": <sample_id>,
    "source": [list of strings, each string is a doc],
    "reference": [list of strings, each string is a reference summary],
}
```

In data/MultiNews we present the procedure for preparing an example dataset.

After the json files are prepared, run the conversion to h5df:
```
mkdir data/MultiNews/h5
cd src

python json2h5.py --input ../data/MultiNews/multinews.train.json \
                  --output ../data/MultiNews/h5/multinews.train.h5df

python json2h5.py --input ../data/MultiNews/multinews.test.json \
                  --output ../data/MultiNews/h5/multinews.test.h5df
```

## Training data

Preparing paired documents for training with the Siamese network
```
python prepare.py ../data/MultiNews/h5/multinews.train.h5df \
                  ../data/MultiNews/h5/multinews.train.paired.h5df \
                  300 \
                  ../data/MultiNews/h5/multinews.vocab.csv

python prepare.py ../data/MultiNews/h5/multinews.test.h5df \
                  ../data/MultiNews/h5/multinews.test.paired.h5df \
                  300 \
                  ../data/MultiNews/h5/multinews.vocab.csv
```

## Model training
```
mkdir -p ../model
python train.py ../data/MultiNews/h5/multinews.train.paired.h5df\
                ../data/MultiNews/h5/multinews.vocab.csv \
                --test_data ../data/MultiNews/h5/multinews.test.paired.h5df \
                --checkpoint ../model/multinews.pt \
                --fasttext_file crawl-300d-2M-subword.bin
```

## Applying a trained model
Application example with leadsum:
```
python docscore_lead.py ../data/MultiNews/h5/multinews.test.h5df \
                        ../model/multinews.pt \
                        300 \
                        ../data/MultiNews/h5/multinews.vocab.csv \
                         --logfile ../logs/multinews.log
```

To keep the summaries produced in the rouge_temp folder:

```
python docscore_lead.py ../data/MultiNews/h5/multinews.test.h5df \
                        ../model/multinews.pt \
                        300 \
                        ../data/MultiNews/h5/multinews.vocab.csv \
                         --logfile ../logs/multinews.log
                         --preserve_summaries
```

We provide the not prioritized version for comparison.
```
python leadsum.py ../data/MultiNews/h5/multinews.test.h5df \
                  300 \
                  --logfile ../logs/multinews.log
```


## Citation

If you find this code useful, please cite:
```
@INPROCEEDINGS{240972,
    AUTHOR="Klaifer Garcia and Lilian Berton",
    TITLE="Siamese Network-Based Prioritization for Enhanced Multi-Document Summarization",
    BOOKTITLE="BRACIS 2024 () ",
    ADDRESS="",
    DAYS="23-21",
    MONTH="may",
    YEAR="2024",
    KEYWORDS="- Natural Language Processing; - Neural Networks",
    URL="http://XXXXX/240972.pdf"
}
```