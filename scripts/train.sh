config_file='experiments/USPTO_50K_Transformer.yaml'

CUDA_VISIBLE_DEVICES=0 torchrun train.py --config $config_file
