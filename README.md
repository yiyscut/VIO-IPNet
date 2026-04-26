<div align="center">
  <h1>A Plug-and-Play Learning-based IMU Bias Factor for Robust Visual-Inertial Odometry</h1>
</div>

<h3 align="center">
    <a href="">Yang Yi</a>, 
    <a href="">Kunqing Wang</a>, 
    <a href="">Jinpu Zhang</a>,
    <a href="">Zhen Tan</a>, 
    <a href="">Xiangke Wang</a>, 
    <a href="">Hui Shen</a><sup>*</sup>, 
    <a href="">Dewen Hu</a>
</h3>

<p align="center">
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
    </a>
    <a href="">
        <img src="https://img.shields.io/badge/Python-3.10+-blue" />
    </a>
    <a href="">
        <img src="https://img.shields.io/badge/Status-TIM Accept-green" />
    </a>
</p>

## Tested Environment

- OS: Ubuntu 20.04.6 LTS
- CUDA: 12.4
- PyTorch: 2.6.0+cu124

<video controls width="600" src="Video/IPNet.mp4"></video>

## Installation

We recommend Python 3.10.

```bash
conda create -n ipnet python=3.10
conda activate ipnet
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/euroc.conf --device cuda:0
```

On the first run, the script generates pseudo labels under `experiments/.../label/` and exits automatically. After that, rerun the script.

## Inference

```bash
python inference.py --config configs/euroc.conf --device cuda:0
```

Predicted IMU bias and noise sequences are saved to `experiments/.../bag/<sequence>/<sequence>.csv`.

## ROS Bag Export

```bash
python createbag.py --config configs/euroc.conf --device cuda:0
```

The exported bag file is written to `experiments/.../bag/<sequence>/<sequence>.bag`.

## Visualize

```bash
python visualize_label.py --config configs/euroc.conf --device cuda:0
```

Figures are saved to `experiments/.../debug/`.

## Acknowledgement

This work is built upon [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion).


## Citation
If you find our work useful, please cite:
```bibtex
@ARTICLE{11455506,
  author={Yi, Yang and Wang, Kunqing and Zhang, Jinpu and Tan, Zhen and Wang, Xiangke and Shen, Hui and Hu, Dewen},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={A Plug-and-Play Learning-Based IMU Bias Factor for Robust Visual–Inertial Odometry}, 
  year={2026},
  volume={75},
  number={},
  pages={1-12},
  doi={10.1109/TIM.2026.3676182}}
```