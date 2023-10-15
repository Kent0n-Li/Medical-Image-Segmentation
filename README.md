# MedSeg: Medical Image Segmentation GUI Toolbox
# 可视化医学图像分割工具箱

## Get all segmentation baselines without writing any code.
## 不用写任何代码便可以运行所有分割模型

Install:

```bash
conda create -n nnsam python=3.9
conda activate nnsam
git clone https://github.com/Kent0n-Li/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation-Benchmark

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install timm
pip install git+https://github.com/Kent0n-Li/nnSAM.git
pip install -r requirements.txt

python web.py
```

If you only want to use nnSAM, please install [this](https://github.com/Kent0n-Li/nnSAM).

![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img1.png)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img2.png)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img3.png)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img4.png)
