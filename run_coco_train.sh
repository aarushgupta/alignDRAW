devices=$1
echo "Running training on GPU $devices"
CUDA_VISIBLE_DEVICES=$devices python train.py --dataset_name coco --input_image_size 32 --n_channels 3 --run_idx 0 --n_epochs 62500 --save_after 1000 --print_after 20 --z_size 275 --read_N 9 --write_N 9 --dec_size 550 --enc_size 550