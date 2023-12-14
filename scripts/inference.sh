results_path="results/USPTO_50K/Retroformer_dim512_wd0.001/2023-12-07-02-06"
model_path=$results_path/saved_model

# pretrained_path=$model_path/model.chkpt
pretrained_path=$model_path/avg.pt

# for num in $(seq 2 20)
# do
python utils/avg.py --inputs $model_path --output $pretrained_path --num-epoch-checkpoints 12

output_path=$results_path/test_result

CUDA_VISIBLE_DEVICES=3 python inference.py --pretrained_path $pretrained_path --output_path $output_path

datasets=$results_path/test_result_gt

python evaluation.py -o $output_path  -t $datasets -c 12 -n 1 
python evaluation.py -o $output_path  -t $datasets -c 12 -n 3 
python evaluation.py -o $output_path  -t $datasets -c 12 -n 5 
python evaluation.py -o $output_path  -t $datasets -c 12 -n 10
# done