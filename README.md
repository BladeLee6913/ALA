# Research on Fracture Recognition in Well Logging Images: Adversarial Learning with Attention

## Overview

This project is PyTorch implementation used in **adversarial learning with attention** in order to perform the extraction of fractures in well logging images. We utilize the CRACK 500 dataset as source images with annotations to achieve unsupervised domain adaptation. The results show the satisfactory performances in fractures recognition.

Code written by Zhipeng Li, University of Electronic Science and Technology of China(UESTC). If you have any queries, please don't hesitate to contact me at lizhipeng1202@qq.com.

## Quick start

### Training

Run main_ALA.py

### Testing

Run Prediction.py

## Environments

PyTorch 1.2

Python 3.7

We train this network based on windows and using CUDA 10.0 to accelerate calculations.

## References

- Yi-Hsuan Tsai, Wei-Chih Hung, Samuel Schulter, Kihyuk Sohn, Ming-Hsuan Yang, Manmohan Chandraker. Learning to Adapt Structured Output Space for Semantic Segmentation. in CVPR 2018. 
- Abhijit Guha Roy, Nassir Navab, Christian Wachinger. Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks. in MICCAI 2018. 
- Abhijit Guha Roy, Nassir Navab, Christian Wachinger. Recalibrating Fully Convolutional Networks With Spatial and Channel “Squeeze and Excitation” Blocks. in *IEEE Transactions on Medical Imaging*, vol. 38, no. 2, pp. 540-549, Feb. 2019.