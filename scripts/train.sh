config_file='experiments/USPTO_50K_Transformer.yaml'

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port='29503' train.py --config $config_file
