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
### Install dependencies.

1. create an  environment

```
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

### Acknowledge

We thank the authors of [Gaussian Splatting](), [Mip-Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main), [RadeGS](https://github.com/BaowenZ/RaDe-GS/tree/main), and the repos for their great works.

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
