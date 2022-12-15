set -e
devices=$1
run_idx=$2
dataset_path=$3

echo "Running training on GPU $devices"
CUDA_VISIBLE_DEVICES=$devices python train.py --dataset_name coco --model_name tti --input_image_size 64 --n_channels 3 --run_idx $run_idx --n_epochs 250 --save_after 50 --log_after 1 --batch_size 128 --T 200 --lr 1e-3 --beta1 0.9 --no_clip_grad --ft_lang --dataset_path $dataset_path
