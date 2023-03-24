
# Unsupervised Class-incremental Learning through Confusion

This repository is the official implementation of Unsupervised Class-incremental Learning through Confusion. 


## Requirements

To install requirements:

```setup
conda env create -f final_environment.yml
```
Also, download the datasets in test and train format from the following drive links:-
* CIFAR10 - https://drive.google.com/file/d/1gAKYFu_B7GkbfonqTYIUKwry2PuaAKO4/view?usp=sharing
* CIFAR100 - https://drive.google.com/file/d/1Yyg12DQCkWKXbwjJUz4LkMqqn9Zijq4c/view?usp=sharing
* MNIST - https://drive.google.com/file/d/102ldR97ivVhCFsY2KqEuf0Ol6I1pPkCe/view?usp=sharing
* SVHN - https://drive.google.com/file/d/1DhrjOZfmBwga_bnptqF5NA0ojLV7oPXW/view?usp=sharing



## Training

To train the model(s) in the paper, run this command:

```train
python train.py --<Dataset Name> --exposure_size <exposure size> --num_exposures <Number of exposures> --create_data --mode <mode_name> --path <path to save model and mapping> --epochs 15 
```

Supported dataset names are - 'cifar10', 'cifar100', 'mnist' and 'svhn'. 
Mode names for training with iLAP or other methods are as follows:
* iLAP with class imbalance - 'iLAP_CI'
* iLAP without class imbalance - 'iLAP_WCI'
* IOLfCV - 'iolfcv'
* Supervised training - 'supervised'

Training saves model and mapping in the path specified.


## Evaluation

To evaluate my model, run:

```eval
python eval.py --<Dataset Name> --model <trained model path> --test_dir <Test directory path> --no_classes <Number of classes for dataset> --map_path <Learned mapping path>
```


## Pre-trained Models

You can download pretrained models for CIFAR10 with 200 exposure size and 100 exposures here -
https://drive.google.com/file/d/1dOEx0qW4c-fy0LTZONXG7TMP3wz2i2Kv/view?usp=sharing


## Results

These are the results achieved by our model compared to other methods:


|                         | MNIST | SVHN\-MNIST | CIFAR\-10 | CIFAR\-100 |
|-------------------------|-------|-------------|-----------|------------|
| **E2EIL \(Supervised\)**    | **98\.1** | **89\.4**       | **74\.7**     | **72\.5**      |
| **IOLfCV \(Unsupervised\)** | 88\.1 | 47\.7       | 45\.5     | 61\.6      |
| **iLAP w/o CI \(Ours\)**    | 90\.5 | 84\.6       | 60\.6     | 65\.0      |
| **iLAP w/ CI \(Ours\)**     | **98\.1** | **88\.2**       | **67\.6**     | **68\.6**      |


