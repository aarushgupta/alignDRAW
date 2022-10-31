devices=$1
run_idx=$2
echo "Running training on GPU $devices"
CUDA_VISIBLE_DEVICES=$devices python train.py --dataset_name mnist_captions --input_image_size 60 --n_channels 1 --run_idx $run_idx --lang_inp_size 23 --z_size 100
