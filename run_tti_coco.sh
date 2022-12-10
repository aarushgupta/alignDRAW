devices=$1
run_idx=$2
echo "Running training on GPU $devices"
CUDA_VISIBLE_DEVICES=$devices python train.py --dataset_name coco --model_name tti --input_image_size 32 --n_channels 3 --run_idx $run_idx --n_epochs 50 --save_after 5 --log_after 1 --batch_size 128 --T 200 --lr 1e-3 --beta1 0.9 --no_clip_grad 
