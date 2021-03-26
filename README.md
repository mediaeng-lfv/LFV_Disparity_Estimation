# Depth estimation from 4D light field videos
We release the code of the "Depth Estimation from 4D Light Field Videos".  

For the details, please follow the links below.  
[**[Project]**](https://mediaeng-lfv.github.io/LFV_Disparity_Estimation/)
[**[Paper]**](https://arxiv.org/abs/2012.03021)
[**[Dataset]**](https://ieee-dataport.org/open-access/sintel-4d-light-field-video-dataset)

## Reference
Takahiro Kinoshita and Satoshi Ono. Depth estimation from 4d light field videos. In International Workshop on Advanced Imaging Technology (IWAIT) 2021, Vol. 11766, p. 117660A. International Society for Optics and Photonics, 2021.
```bibtex
@inproceedings{kinoshita2021depth,
  title={Depth estimation from 4D light field videos},
  author={Kinoshita, Takahiro and Ono, Satoshi},
  booktitle={International Workshop on Advanced Imaging Technology (IWAIT) 2021},
  volume={11766},
  pages={117660A},
  year={2021},
  organization={International Society for Optics and Photonics}
}
```

# Network Architecture
![architecture](https://user-images.githubusercontent.com/37448236/101273325-87c0fc00-37d7-11eb-9951-4542e7cc4d95.png)

# Setup
## Environment
- CUDA Toolkit 10.1 update2
- cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1
- Python 3.6.6 (anaconda3)
- tensorflow 2.3.0
- keras 2.4.3

```sh
conda create -n LF_video python=3.6 anaconda
conda activate LF_video
pip install tensorflow==2.3.0 keras==2.4.3
```

## File structure 
Please set up the file structure as follows.  
Download [Sintel_LFV_cross-hair.zip](https://ieee-dataport.org/open-access/sintel-4d-light-field-video-dataset) and unzip it to Sintel_LF.
```
LFV_Disparity_Estimation/
  ┣━━ README.md    ...    this document
  ┣━━ src/    ...    source codes
  ┣━━ Sintel_LF/    ...    downloaded full dataset
  ┃     ┣━━ ambushfight_1/
  ┃     ┣━━ thebigfight_1/
  ┃     ┣━━ .../
  ┣━━ patch_data_fl5/    ...    dir for patch data (the data will be created later.)
  ┃     ┣━━ train_data.txt        ...    scenes list for training
  ┃     ┣━━ validation_data.txt   ...    scenes list for validation
  ┃     ┗━━ test_data.txt         ...    scenes list for test
  ┗━━ output/    ...    dir for output
```

## Code details
```
LFV_Disparity_Estimation/
  ┗━━ src/
        ┣━━ create_dataset.py    ...    create patch data for patch-wise training
        ┃
        ┣━━ models/
        ┃     ┣━━ modules/
        ┃     ┣━━ LFI_conv3D.py   ...    baseline architecture model
        ┃     ┗━━ LFV_conv3D_STCLSTM.py         ...    proposed architecture model
        ┣━━ loss.py    ...    loss function
        ┣━━ sobel.py    ...    sobel filter for loss function
        ┣━━ mygenerator.py    ...    train/validation/test generator
        ┣━━ train.py    ...    training main script
        ┣━━ train_STCLSTM.py    ...    baseline model training
        ┣━━ train_baseline.py    ...    proposed model training
        ┃
        ┣━━ metrics.py    ...    metrics for evaluation
        ┗━━ evaluate.py    ...    evaluate model
```

# Train and evaluate
Clone this repository.
```sh
git clone https://github.com/mediaeng-lfv/LFV_Disparity_Estimation.git
cd LFV_Disparity_Estimation/src
```
Create patch dataset (The first time only.)  
**Note**: The size of all patch data created is 222.5 GiB.  
```sh
python ./create_dataset.py
```
Start training.
```sh
# python ./train_baseline.py
python ./train_STCLSTM.py
```
Evaluate model.
```sh
python evaluate.py MODEL_WEIGHT_PATH
# e.g. MODEL_WEIGHT_PATH = ../output/2020-10-22_0139_STCLSTM/weights.h5
#      MODEL_WEIGHT_PATH = ../output/2020-10-21_1213_baseline/weights.h5
```
