set -e
devices=$1
run_idx=$2
echo "Running validation on GPU $devices"

# Generate image from a random sample from COCO validation dataset
# CUDA_VISIBLE_DEVICES=$devices python visualize_val.py --dataset_name coco --model_name tti --input_image_size 32 --n_channels 3 --run_idx $run_idx --n_epochs 50 --T 200

# Generate image from a custom input sentence
CUDA_VISIBLE_DEVICES=$devices python visualize_val.py --dataset_name coco --model_name tti --input_image_size 32 --n_channels 3 --run_idx $run_idx  --T 200 --custom_sentence "A dog chasing after a human" --trained_model ./results/coco_tti/0/checkpoint/model_final_49

# Sample commands
# CUDA_VISIBLE_DEVICES=$devices python visualize_val.py --dataset_name coco --model_name tti --input_image_size 32 --n_channels 3 --run_idx 0  --T 200 --custom_sentence "Bakery shop selling muffins"
# CUDA_VISIBLE_DEVICES=$devices python visualize_val.py --dataset_name coco --model_name tti --input_image_size 32 --n_channels 3 --run_idx 3  --T 200 --custom_sentence "Bakery shop selling muffins"
# CUDA_VISIBLE_DEVICES=$devices python visualize_val.py --dataset_name coco --model_name tti --input_image_size 64 --n_channels 3 --run_idx 2  --T 200 --custom_sentence "Bakery shop selling muffins"
# CUDA_VISIBLE_DEVICES=$devices python visualize_val.py --dataset_name coco --model_name tti --input_image_size 64 --n_channels 3 --run_idx 4  --T 200 --custom_sentence "Bakery shop selling muffins"
# CUDA_VISIBLE_DEVICES=$devices python visualize_val.py --dataset_name coco --model_name tti --input_image_size 64 --n_channels 3 --run_idx 4 --n_epochs 200 --T 200 --custom_sentence "Chef serving margarita pizza"