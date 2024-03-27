# Fitting Autoregressive Graph Generative Models through Maximum Likelihood Estimation


This repository contains PyTorch implementation of the submission of jmlr: [Fitting Autoregressive Graph Generative Models
through Maximum Likelihood Estimation](https://www.jmlr.org/papers/volume24/22-0337/22-0337.pdf)

This is a minimum working version of the code used for the paper.
## 0. Environment Setup
enviroment setup 
```
pip install -r requirements.txt
```
Run build.sh script in the project's root directory for MMD computation.
```
./build.sh
```




## 1. Training
To list the arguments, run the following command:
```
python main.py -h
```
To train the model on datasets with Rout and DAGG, run the following:
```
python main.py -dataset caveman_small
```

## 2.Evaluation

To evaluate the generated graph, run the following:
```
python main.py -task evaluate -load_model_path 'saved_model_path'
```



