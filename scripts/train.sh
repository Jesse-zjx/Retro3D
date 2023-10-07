config_file='experiments/USPTO_50K_Transformer.yaml'

OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port='29501' train.py --config $config_file
