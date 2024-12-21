# Preparing the MultiNews dataset

We performed experiments with [Multinews dataset](https://github.com/Alex-Fabbri/Multi-News).

This dataset contains news and reference summary.
To apply it to our code, we start by converting it to json format.

To do this, download the files. In our experiments we used the "raw data" version and saved it in a folder called raw.

Then run the conversion:

```
cd data/MultiNews/src

python multinews.py --input_src ../raw_version/train.raw.src \
                    --input_tgt ../raw_version/train.raw.tgt \
                    --output ../multinews.train.json 
                    
python multinews.py --input_src ../raw_version/test.raw.src \
                    --input_tgt ../raw_version/test.raw.tgt \
                    --output ../multinews.test.json
```


