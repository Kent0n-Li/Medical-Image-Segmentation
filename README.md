# Medical Image Segmentation Benchmark Toolbox


Install:

```bash
conda create -n nnsam python=3.9
conda activate nnsam
git clone https://github.com/Kent0n-Li/Medical-Image-Segmentation-Benchmark.git
cd Medical-Image-Segmentation-Benchmark

pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install timm
pip install git+https://github.com/Kent0n-Li/nnSAM.git
pip install -r requirements.txt

python web.py
```
