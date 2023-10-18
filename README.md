# MedSeg: Medical Image Segmentation GUI Toolbox <br> 可视化医学图像分割工具箱

## Get all segmentation baselines without writing any code. <br> 不用写任何代码便可以运行所有分割模型

Install (安装步骤):

```bash
conda create -n nnsam python=3.9
conda activate nnsam
```

Choose a suitable Pytorch with CUDA to install <br> 
根据CUDA选择合适版本的Pytorch进行安装
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install timm
pip install git+https://github.com/Kent0n-Li/nnSAM.git

git clone https://github.com/Kent0n-Li/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation-Benchmark
pip install -r requirements.txt
```

运行可视化软件
```bash
python web.py
```

If you only want to use nnSAM, please install [this](https://github.com/Kent0n-Li/nnSAM). <br>
如果你只想运行nnSAM,请访问该代码仓库[this](https://github.com/Kent0n-Li/nnSAM)

![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img1.png)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img2.png)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img3.png)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img4.png)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img5.png)



## Citation

If you find this repository useful for your research, please use the following.

```
@article{li2023nnsam,
  title={nnSAM: Plug-and-play Segment Anything Model Improves nnUNet Performance},
  author={Li, Yunxiang and Jing, Bowen and Li, Zihan and Wang, Jing and Zhang, You},
  journal={arXiv preprint arXiv:2309.16967},
  year={2023}
}
```
