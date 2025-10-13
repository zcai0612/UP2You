<h2 align="center">
 
  <a href="https://zcai0612.github.io/UP2You/">UP2You: Fast Reconstruction of Yourself from Unconstrained Photo Collections</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.06219-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.24817) 
[![Home Page](https://img.shields.io/badge/Project-Website-C27185.svg)](https://zcai0612.github.io/UP2You/) 
[![X](https://img.shields.io/badge/@Zeyu%20Cai-black?logo=X)](https://x.com/ZeyuCai111168)

[Zeyu Cai](https://zcai0612.github.io/),
[Ziyang Li](https://github.com/Ziyang-Li-AILab/),
[Xiaoben Li](https://xiaobenli00.github.io/),
[Boqian Li](https://boqian-li.github.io/),
[Zeyu Wang](https://cislab.hkust-gz.edu.cn/members/zeyu-wang/),
[Zhenyu Zhang](https://jessezhang92.github.io/)†,
[Yuliang Xiu](https://xiuyuliang.cn/)†

</h5>

<div align="center">
TL;DR: Tuning-free, Fast Reconstruct Yourself from Unconstrained Photo Collections
</div>
<br>

<div align="center">
    <img src="assets/teaser.gif" alt="UP2You Teaser" autoplay loop>
</div>


## Getting Started

### Installation

1. Clone UP2You.
```bash
git clone https://github.com/zcai0612/UP2You.git
cd UP2You
```

2. Create the environment.
```bash
conda create -n up2you python=3.10
conda activate up2you

# torch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# kaolin
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu118.html

pip install -r requirements.txt

# https://github.com/cnr-isti-vclab/meshlab/issues/1461
conda install -y libffi==3.3

# install pytorch3d, download from conda
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt241.tar.bz2
conda install pytorch3d-0.7.8-py310_cu118_pyt241.tar.bz2
```

### Download
Download `pretrained_models` and `human_models` from huggingface [`Co2y/UP2You`](https://huggingface.co/Co2y/UP2You), and put them into the project directory. You can refer to following commands:
```Bash
export HF_ENDPOINT="https://hf-mirror.com" # Optional

hf download Co2y/UP2You --local-dir ./src # download models

mv ./src/human_models ./
mv ./src/pretrained_models ./

rm -rf ./src
```

### Inference
To run the inference pipeline, you can use the following command:
```bash
python inference_low_gpu.py \
    --base_model_path stabilityai/stable-diffusion-2-1-base  \
    --segment_model_name ZhengPeng7/BiRefNet \
    --data_dir examples \
    --output_dir outputs \
```

or you can just use `run.sh`:
```bash
bash run.sh 
```

Here we provide an example, where `examples` is the folder of unconstrained photos and `outputs` is the output directory of generated results.


## Acknowledgements
Our code is based on the following awesome repositories and datasets:

- [MV-Adapter](https://github.com/huanngzh/MV-Adapter), [PuzzleAvatar](https://github.com/YuliangXiu/PuzzleAvatar), [VGGT](https://github.com/facebookresearch/vggt), [PSHuman](https://github.com/pengHTYX/PSHuman), [SOAP](https://github.com/TingtingLiao/soap)
- [THuman2.1](https://github.com/ytrock/THuman2.0-Dataset), [CustomHumans](https://custom-humans.github.io/), [2k2k](https://github.com/SangHunHan92/2K2K), [Human4DiT](https://github.com/DSaurus/Human4DiT), [4D-Dress](https://github.com/eth-ait/4d-dress), [PuzzleIOI](https://puzzleavatar.is.tue.mpg.de/)



We thank the authors for releasing their code and data !

We thank [Siyuan Yu](https://ysysimon.com/home) for the help in Houdini Simulation, [Shunsuke Saito](https://shunsukesaito.github.io/), [Dianbing Xi](https://scholar.google.com/citations?user=7H29mf4AAAAJ&hl=zh-CN&scioq=UP2You&oi=sra), [Yifei Zeng](https://zeng-yifei.github.io/) for the fruitful discussions, and the members of [Endless AI Lab](https://xiuyuliang.cn/group.html) for their help on data capture and discussions.

## Citation

If you find our work useful, please cite:

```bibtex
@article{cai2025up2you,
  title={UP2You: Fast Reconstruction of Yourself from Unconstrained Photo Collections},
  author={Cai, Zeyu and Li, Ziyang and Li, Xiaoben and Li, Boqian and Wang, Zeyu and Zhang, Zhenyu and Xiu, Yuliang},
  journal={arXiv preprint arXiv:2509.24817},
  year={2025}
}
```