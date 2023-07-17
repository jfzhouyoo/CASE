
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
@inproceedings{DBLP:conf/acl/ZhouZW0H23,
  author       = {Jinfeng Zhou and
                  Chujie Zheng and
                  Bo Wang and
                  Zheng Zhang and
                  Minlie Huang},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {{CASE:} Aligning Coarse-to-Fine Cognition and Affection for Empathetic
                  Response Generation},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2023, Toronto, Canada,
                  July 9-14, 2023},
  pages        = {8223--8237},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.acl-long.457},
  timestamp    = {Thu, 13 Jul 2023 16:47:40 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/ZhouZW0H23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


