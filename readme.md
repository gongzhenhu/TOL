## Improving Aspect-Based Sentiment Analysis via Tuple-Order Learning

This repository contains the PyTorch implementation of our ECAI 2024 paper [Improving Aspect-Based Sentiment Analysis via Tuple-Order Learning](https://www.researchgate.net/publication/385027572_Improving_Aspect-Based_Sentiment_Analysis_via_Tuple-Order_Learning). Our code is modified from [MVP](https://github.com/ZubinGou/multi-view-prompting).

### Environment Setup and Data Preprocessing
```sh
conda create -n TOL python=3.8
conda activate TOL
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Training and Evaluation
Train and Evaluation
```text
bash scripts/run_ada.sh
```

### Statement
When training the tuple evaluation model, the uploaded code saves two models at the end. If both models consider the tuples as simple tuples, we classify them as simple tuples; otherwise, we sort them according to loss.

### Citation
If you find this code helpful for your research, please consider citing
```text
    @inproceedings{DBLP:conf/ecai/HuLZZ24,
    author       = {Gongzhen Hu and
                    Yuanjun Liu and
                    Xiabing Zhou and
                    Min Zhang},
    title        = {Improving Aspect-Based Sentiment Analysis via Tuple-Order Learning},
    booktitle    = {{ECAI} 2024 - 27th European Conference on Artificial Intelligence,
                    19-24 October 2024, Santiago de Compostela, Spain - Including 13th
                    Conference on Prestigious Applications of Intelligent Systems {(PAIS}
                    2024)},
    volume       = {392},
    pages        = {3972--3979},
    publisher    = {{IOS} Press},
    year         = {2024},
    url          = {https://doi.org/10.3233/FAIA240963},
    }

```