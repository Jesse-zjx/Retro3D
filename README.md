## Summary of Retro3D retrosynthetic model
The current repository contains Retro3D retrosynthetic models.

## Environment Preparation
``` bash
conda create -n retro3d python==3.9
conda activate retro3d
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -r requirements.txt
```

## Data
Download the raw reaction dataset from [here](https://github.com/Hanjun-Dai/GLN) and put it into your data directory (e.g., `datasets/data/USPTO_50K`). One can also create your own reaction dataset as long as the data shares the same format (columns: `id`, `class`, `reactants>reagents>production`) and the reactions are atom-mapped.

## Train a model
All experiment-related parameters are written in the configuration file (e.g. ./experiments/USPTO_50K_Retro3D.yaml)

All outputs (log files and checkpoints) will be saved to the working directory, which is specified by OUTPUT_DIR in the config file.

### Train by DistributedDataParallel(DDP)
```shell
torchrun --nproc_per_node=1 train.py --config ${CONFIG_FILE}
```

## Average checkpoints
```shell
python tools/avg_all.py --inputs ${CHECKPOINT_FILE_PATH} --output ${OUTPUT_FILE_PATH} --num-epoch-checkpoints ${NUM_EPOCH_CHECKPOINTS} --checkpoint-upper-bound ${CHECKPOINTS_UPPERBOUND}
```
```shell
eg:
python utils/avg.py --inputs results/USPTO_50K/Retro3D_dim512_wd0.001/2023-12-07-02-06/saved_model --output results/USPTO_50K/Retro3D_dim512_wd0.001/2023-12-07-02-06/saved_model/avg.pt --num-epoch-checkpoints 7 --checkpoint-upper-bound 149
```

## Inference with pretrained models
```shell
python inference --pretrained_path ${CHECKPOINT_FILE} --output_path ${RESULT_FILE}
```
```shell
eg:
python inference.py --pretrained_path results/USPTO_50K/Retro3D_dim512_wd0.001/2023-12-07-02-06/saved_model/avg.pt --output_path results/USPTO_50K/Retro3D_dim512_wd0.001/2023-12-07-02-06/test_result
```

## Evaluaction inference result
```shell
python evaluaction.py -o ${INFERENCE_OUTPUT_FILE} -t ${TESTSET_TARGET_FILE_PATH} -n ${TOP_N} -c ${NUM_CORES}
```
```shell
eg:
python evaluation.py -o results/USPTO_50K/Retro3D_dim512_wd0.001/2023-12-07-02-06/test_result  -t results/USPTO_50K/Retro3D_dim512_wd0.001/2023-12-07-02-06/test_result_gt -c 12 -n 1 
```