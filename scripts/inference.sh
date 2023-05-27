results_path="results/USPTO_50K_AM/Retroformer_dim256_wd0.0/2023-05-22-23-27"
model_path=$results_path/saved_model

# pretrained_path=$model_path/model.chkpt
pretrained_path=$model_path/avg.pt

python utils/avg.py --inputs $model_path --output $pretrained_path --num-epoch-checkpoints 7

output_path=$results_path/test_result

CUDA_VISIBLE_DEVICES=0 python inference.py --pretrained_path $pretrained_path --output_path $output_path

datasets=$results_path/test_result_gt

python evaluation.py -o $output_path  -t $datasets -c 12 -n 1 
python evaluation.py -o $output_path  -t $datasets -c 12 -n 3 
python evaluation.py -o $output_path  -t $datasets -c 12 -n 5 
python evaluation.py -o $output_path  -t $datasets -c 12 -n 10