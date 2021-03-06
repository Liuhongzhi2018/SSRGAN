# SSRGAN for Spectral Reconstruction from RGB

## Prerequisites
- Ubuntu 16.04
- Python 3.7
- PyTorch 1.1
- NVIDIA GPU (8G memory or larger) + CUDA cuDNN

## Download data
[NTIRE 2020 Spectral Reconstruction Challenge - Track 1: Clean](https://competitions.codalab.org/competitions/22225).

- Place the data at *NTIRE2020* and arrange the directories as follows:

    *NTIRE2020/NTIRE2020_Train_Spectral*  
    --ARAD_HS_0001.mat  
    ......  
    --ARAD_HS_0450.mat  
    
    *NTIRE2020/NTIRE2020_Train_Clean*  
    --ARAD_HS_0001_clean.png   
    ......  
    --ARAD_HS_0450_clean.png  
    
    *NTIRE2020/NTIRE2020_Validation_Spectral*  
    --ARAD_HS_0451.mat   
    ......  
    --ARAD_HS_0465.mat  
    
    *NTIRE2020/NTIRE2020_Validation_Clean*  
    --ARAD_HS_0451_clean.png  
    ......  
    --ARAD_HS_0465_clean.png  


## Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate 
```
- Clone this repo:
```bash
git clone https://github.com/Liuhongzhi2018/SSRGAN.git
cd SSRGAN
```

## Train and validate the model
- If you are running the code for the first time, remember to run data pre-processing firstly.
```bash
python data_preprocess.py --mode train
python data_preprocess.py --mode val
```    
- Otherwise, run training directly.
```bash
python main.py --gpu_ids=1 --name RGB2HSI_1028
python main.py --gpu_ids=0 --name RGB2HSI_1108_long --niter=139 --end_epoch=300
``` 
- If you are trying to use local enhance model
```bash 
python main.py ----gpu_ids=0 --name RGB2HSI_1108 --netG=local --n_blocks_global=4

python main.py ----gpu_ids=5 --name RGB2HSI_1120 --netG=local --n_blocks_global=5 n_blocks_local=1 --niter=150

python main.py --gpu_ids=5 --name RGB2HSI_1120 --niter=150

# Server 159
python main.py --gpu_ids=5 --name RGB2HSI_1120 --netG=local --n_blocks_global=6 --n_blocks_local=1  --niter=150

# OUC Lab
python main.py --gpu_ids=0 --name RGB2HSI_1205 --trade_off 10

python main.py --gpu_ids=1 --name RGB2HSI_12051 --trade_off 1
```

## Continue to train the model
- If you want to fine the pretrained model.
```
python main.py
 --gpu_ids=1 --name RGB2HSI_1028 --continue_train --end_epoch 500
```

## Test the model

To be continued
