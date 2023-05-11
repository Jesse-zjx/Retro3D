config_file='experiments/USPTO_50K_Transformer.yaml'

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port='29501' train.py --config $config_file
