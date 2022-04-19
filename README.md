# SAPCU
【Code of CVPR 2022 paper】 

Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation

Paper address: https://arxiv.org/abs/2204.08196

## Environment
Pytorch 1.9.0

CUDA 10.2

## Evaluation
### a. Download models
Download the pretrained models from the link and unzip it to  `./out/`
```
https://pan.baidu.com/s/1OPVnCHq129DBMWh5BA2Whg 
access code: hgii 
```
### b. Compilation
Run the following command for compiling dense.cpp which generates dense seed points
```
g++ -std=c++11 dense.cpp -O2 -o dense
```
### c. Evaluation
You can now test our code on the provided point clouds in the `test` folder. To this end, simply run
```
python generate.py
```
The 4X upsampling results will be created in the `testout` folder.

## Training
Download the training dataset from the link and unzip it to the working directory
```
https://pan.baidu.com/s/1VQ-3RFO02fQfcLBfqvCBZA 
access code: vpfm 
```

Then run the following commands for training our network
```
python trainfn.py
python trainfd.py
```

## Citation
If the repo is useful for your research, please consider citing:
  
    @inproceedings{sapcu,
      title = {Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation},
      author = {Wenbo Zhao, Xianming Liu, Zhiwei Zhong, Junjun Jian, Wei Gao, Ge Li, Xiangyang Ji},
      booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      year = {2022} 
    }


## Acknowledgement
The code is based on [occupancy_networks](https://github.com/autonomousvision/occupancy_networks/) and [DGCNN](https://github.com/WangYueFt/dgcnn), If you use any of this code, please make sure to cite these works.
