# stock-market-forecasting

## Prerequisites
Before you start using scripts in this repository you have to configure your workspace. We've limited ourselves to only 
use Python 3. So you have to be sure that you've installed it. Some of the requirements are listed in `requirements.txt`
file. These you can install with `pip` or `conda`. Moreover:
* make sure that all the data downloaded from [Kaggle](https://www.kaggle.com/aaron7sun/stocknews) are unpacked in 
`./data/raw` directory,
* the information about GloVe model that we've used in modelling can be found 
[here](https://spacy.io/models/en#section-en_core_web_lg),
* for running RNN models you have to install [cuDNN](https://developer.nvidia.com/cudnn) or replace `CuDNNGRU` layers
with plain old `GRU` (slower).

## How to work with this repository?

All scripts should be run from the repository's root directory!

1. Download and unzip all raw data from [Kaggle](https://www.kaggle.com/aaron7sun/stocknews) to `./data/raw`.
2. Run `./scripts/data/change_classes.py` to assign labels to dataset based on stock open and close value.
3. Prepare data used for corpus cleaning by running `./scripts/data/prepare_*.py` and than clean corpus with 
`./scripts/clean_corpus.py`.
4. If you intend to run a GloVe or Doc2Vec model you must first run a proper script from `./scripts/features` 
directory.
5. Now you should be able to run scripts from `./experiments` and `./scripts/model` directories.