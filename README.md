<p align="center">
  <h1 align="center">DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://leoshen917.github.io/"><strong>Liao Shen</strong></a>
    &nbsp;&nbsp;
    <a href="https://tqtqliu.github.io/"><strong>Tianqi Liu</strong></a>
    &nbsp;&nbsp;
    <a href="https://huiqiang-sun.github.io/"><strong>Huiqiang Sun</strong></a>
    &nbsp;&nbsp;
    <a href="https://lijia7.github.io"><strong>Jiaqi Li</strong></a>
    &nbsp;&nbsp;
    <a href="http://english.aia.hust.edu.cn/info/1085/1528.htm"><strong>Zhiguo Cao</strong></a>
    &nbsp;&nbsp;
    <a href="https://weivision.github.io/"><strong>Wei Li<sep>✉</sep></strong></a>
    &nbsp;&nbsp;
    <a href="https://www.mmlab-ntu.com/person/ccloy/index.html"<strong>Chen Change Loy</strong></a>
  </p>
  <p align="center">
    <a href="https://dof-gaussian.github.io/"<strong>Project Page</strong></a> |<a href="https://arxiv.org/abs/2503.00746"<strong>Arxiv</strong></a> |<a href="https://www.youtube.com/watch?v=-kaWXVW0TCg"<strong>Video</strong></a> 
  </p>
  <p align="center">
    CVPR 2025
  </p>
</p>


<div align="center">
        <img src="./assets/teaser1.png">
  </div>

### Install dependencies.

1. Clone the repository and create an  environment.

```
git clone https://github.com/leoShen917/DoF-Gaussian.git --recursive 
conda activate -n dofgs python=3.9
conda activate dofgs
```

2. install pytorch and other dependencies.

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install -r requirements.txt
```

- We use COLMAP to calculate poses and sparse depths. However, original COLMAP does not have fusion mask for each view. Thus, we add masks to COLMAP and denote it as a submodule. Please follow https://colmap.github.io/install.html to install COLMAP in `./colmap` folder (Note that do not cover colmap folder with the original version).
- Considering that the COLMAP repository linked to seems to be missing some of the required files, I've uploaded a version on Baidu disk.
  链接：https://pan.baidu.com/s/1mjlHwC4O6FrekbNguDaMDQ 
  提取码：2mi8

### Preparation

Please download datasets at [here](https://drive.google.com/drive/folders/1qXSgGWUbgIfKdNK16AytEHvxO0lRZ0K5). This dataset is originally produced by [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF). You can organize your own dataset as:

```
real_defocus_blur
│
└─── defocuscupcake
│   │
|   └─── sparse
│   │
|   └─── images_4
│   │
|   └─── images
│   │
|   └─── hold = 8
│ 
└─── defocuscups
│   │
......

```

**Run Colmap**

```
sh colmap.sh <path to dataset>
```

### Training

```
python train.py -s <path to dataset> -m <output folder> --eval -r <downsample_res> --llffhold <llffhold>

// ex. python train.py -s real_defocus_blur/defocuscupcake -m output/defocuscupcake --eval -r 4 --llffhold 8
```

### Evaluation

To calculate metrics, namely PSNR, SSIM, and LPIPS, we could execute the following code:

```
python metric.py -m <output folder> -r <downsample_res>
```

If testing on Deblur-NeRF datasets, we should also add `-t deblur-nerf` at the end of the code chunk above.

### Applications

(coming soon)

### Acknowledge

We thank the authors of [Gaussian Splatting](), [Mip-Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main), [RadeGS](https://github.com/BaowenZ/RaDe-GS/tree/main), [DoF-NeRF](https://github.com/zijinwuzijin/DoF-NeRF), and the repos for their great works.

### Citation
If you find our work useful in your research, please consider to cite our paper:

```
@article{shen2025dof,
  title={DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting},
  author={Shen, Liao and Liu, Tianqi and Sun, Huiqiang and Li, Jiaqi and Cao, Zhiguo and Li, Wei and Loy, Chen Change},
  journal={arXiv preprint arXiv:2503.00746},
  year={2025}
}
```
