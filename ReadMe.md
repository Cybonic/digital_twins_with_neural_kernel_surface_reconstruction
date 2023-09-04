# Recreate High-Fidelity Digital Twins with Neural Kernel Surface Reconstruction from NVIDIA 

https://developer.nvidia.com/blog/recreate-high-fidelity-digital-twins-with-neural-kernel-surface-reconstruction/?ncid=so-link-508891-vt37&=&linkId=100000214326019#cid=_so-link_en-us

### Dependencies 
- Cuda 11.8.0
- Pytorch 2.0
- Python 3.10
- open3d 0.17.0

### Conda environment
```
conda activate digitaltweens
```

# Installation Guide
## Install Cuda
```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
##  Install Pytroch 2.0
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### Check if pytorch is installed
```
python
````
In the python environment run:
```
>>> import torch
>>> torch.cuda.is_available()
true
```
## Install NKSR

Install nksr repository
```
pip install nksr -f https://nksr.s3.ap-northeast-1.amazonaws.com/whl/torch-2.0.0%2Bcu118.html
````
Install dependencies

```
pip install torch_scatter open3d
pip install python-pycg[full] -f https://pycg.s3.4
ap-northeast-1.amazonaws.com/packages/index.html
````

## Deactivate Environment
````
conda deactivate
`````
## Viz mesh
````
python script_viz_ply.py  
````

