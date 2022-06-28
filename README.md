# ATST-RCT: For DCASE 2022 task4 challenge.

This is the official implementation of ATST-RCT.

## Introduction

ATST is a self-supervised pretraining model designed for clip-level audio tasks. Please refer to [ATST official page](https://github.com/Audio-WestlakeU/audiossl) for more information.

RCT is a semi-supervised learning scheme designed for sound event detection. Please refer to [RCT official page](https://github.com/Audio-WestlakeU/RCT) for more informaiton.

## Training

The training/validation data is obtained from the DCSAE2022 task4 [DESED dataset](https://github.com/turpaultn/DESED).
The download of DESED is quite tedious and not all data is available for the accesses. You could ask for help from the DCASE committee to get the full dataset. Noted that, your testing result might be different with an incomplete validation dataset.

To train the model, please first get the baseline architecture of [DCASE2022 task 4](https://github.com/DCASE-REPO/DESED_task)
by:
```bash
git clone git@github.com:DCASE-REPO/DESED_task.git
```
Don't forget to configure your environment by their requirements. And install any packages required. Dont't forget to change the path of the dataset to your owns.

Then, please cover the official DESED repo with ATST-RCT codes in this repo and run:
```bash
python train_fusion_rct.py
```

## Results
The result of the challenge is not published, please refer to their [official page](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments).



## Reference
[1] DESED Dataset: https://github.com/turpaultn/DESED

[2] DCASE2022 Task4 baseline: https://github.com/DCASE-REPO/DESED_task

[3] FilterAug: https://github.com/frednam93/FilterAugSED
