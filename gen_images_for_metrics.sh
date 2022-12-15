set -u
devices=$1
run_idx=$2
pretrained_model=$3

# Example command
# CUDA_VISIBLE_DEVICES=$devices python generate_eval_images.py --dont_encode_text --dont_transform_image --run_idx $run_idx --input_image_size 32 --trained_model ./results/coco_tti/0/checkpoint/model_final_49

CUDA_VISIBLE_DEVICES=$devices python generate_eval_images.py --dont_encode_text --dont_transform_image --run_idx $run_idx --input_image_size 32 --trained_model $pretrained_model