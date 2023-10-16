## Summary of Retro3D retrosynthetic model
The current repository contains Retro3D retrosynthetic models.

## Environment Preparation
``` bash
conda create -n retro3d python==3.7.15
conda activate retro3d
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install -r requirements.txt
```