<p align="center">
  <img width="70%" src="https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/medseglogo.png">
</p>



# MedSeg: Medical Image Segmentation GUI Toolbox <br> 可视化医学图像分割软件

## Get all segmentation baselines without writing any code. <br> 不用写任何代码便可以运行所有分割模型

The current code is designed for Windows, not for Linux. 

If you only want to use nnSAM, please install [this](https://github.com/Kent0n-Li/nnSAM). <br>
如果你只想运行nnSAM,请访问该代码仓库[this](https://github.com/Kent0n-Li/nnSAM)

样例数据集：[Demo Dataset](https://github.com/Kent0n-Li/Medical-Image-Segmentation/tree/main/Demo_dataset)


Install (安装步骤):

```bash
git clone https://github.com/Kent0n-Li/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation-Benchmark
pip install -r requirements.txt
```

Running visualization software (运行可视化软件)
```bash
python web.py
```




## Overview 页面总览
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img1.png)

## Choose Model  选择模型
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img2.png)

## Import Data 导入你的数据集 （2D: png, 3D: nii.gz) 
### 样例数据集：[Demo Dataset](https://github.com/Kent0n-Li/Medical-Image-Segmentation/tree/main/Demo_dataset)
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img3.png)

## Full Auto 全自动模式，一键完成从数据预处理到训练测试和结果总结
![image](https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark/blob/main/img/img4.png)

## Result Summary 结果总结
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
