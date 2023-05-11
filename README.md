
# Code for CASE
> **The code will be released.**

## Requirements

+ Python==3.6.12
+ torch==1.3.0+cu100
+ nltk==3.4.5
+ transformers==4.10.2
+ vaderSentiment==3.3.2
+ tensorboardX==2.5
+ scikit-learn==0.24.1
+ spacy==3.1.4
+ numpy==1.19.5
+ Download  `Pretrained GloVe Embeddings` (`300d`) and save it in `/vectors`.

## Dataset
> **The preprocessed dataset will be released.**

+ The **preprocessing details** can be found in `src/utils/data/loader.py`.
+ If you want to **create the dataset yourself**, download the `comet-atomic-2020` (`BART`) checkpoint and place it in `/data/Comet`. The preprocessed data will be automatically generated after running the `main.sh` script.

## Training, Testing & Evaluation

`bash main.sh`

