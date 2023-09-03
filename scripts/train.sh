config_file='experiments/USPTO_50K_Transformer.yaml'

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config $config_file
