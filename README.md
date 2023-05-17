
# Code for CASE
> The implementation of our paper accepted by ACL 2023: [**CASE: Aligning Coarse-to-Fine Cognition and Affection for Empathetic Response Generation**](https://arxiv.org/abs/2208.08845)

<img src="https://img.shields.io/badge/Venue-ACL--23-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Last%20Updated-2023--05-2D333B" alt="update"/>

## Requirements

+ `Python==3.6.12`
+ `torch==1.3.0+cu100`
+ `nltk==3.4.5`
+ `transformers==4.10.2`
+ `vaderSentiment==3.3.2`
+ `tensorboardX==2.5`
+ `scikit-learn==0.24.1`
+ `spacy==3.1.4`
+ `numpy==1.19.5`
+ Download[`Pretrained GloVe Embeddings 300d`](http://nlp.stanford.edu/data/glove.6B.zip) and save it in `/vectors`.

## Dataset

+ The preprocessed dataset is already provided at [Google Driven](https://drive.google.com/drive/folders/1OUHF7mIxeJwN3jcpYnABKlhPtb_jQzP7?usp=share_link). Change the folder name to `data`.
+ If you want to **create the dataset yourself**, download the [`comet-atomic-2020 (BART) checkpoint`](https://github.com/allenai/comet-atomic-2020) and place it in `/data/Comet`. The preprocessed data will be automatically generated after running the `main.sh` script.

## Training, Testing & Evaluation

```
bash main.sh
```

## Citation

If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{case-zhou,
  author = {Jinfeng Zhou and
            Chujie Zheng and
            Bo Wang and
            Zheng Zhang and
            Minlie Huang},
  title = {{CASE:} Aligning Coarse-to-Fine Cognition and Affection for Empathetic Response Generation},
  booktitle = {ACL},
  year = {2023}
}
```


