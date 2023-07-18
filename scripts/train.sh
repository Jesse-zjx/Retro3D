config_file='experiments/USPTO_50K_Transformer.yaml'

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train.py --config $config_file
