import os
import time
import torch
from tqdm import tqdm

import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import json

from diffusion_model import DDPM
from dataloader import get_data, tokenize_labels
from utils import (
    generate_image,
    get_train_parser,
    plot_sample_images,
    plot_training_losses,
    get_validation_loss,
)
from transformers import RobertaTokenizer

reverse_transform = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.0),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ]
)

def main():

    args = get_train_parser()
    args.trained_model = "./results/coco/tti/1/checkpoint/model_final_99"
    args.dont_encode_text = True
    args.dont_transform_image = True
    args.input_image_size = 64
    args.dataset_name = "coco"
    args.model_name =  "tti"
    args.n_channels = 3
    args.run_idx = 0
    args.T = 200

    # Use GPU is available else use CPU.
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Fetch train and validation loaders
    train_loader, val_loader = get_data(args)

    sample_imgs = next(iter(train_loader))[0]

    args.n_channels = (
        sample_imgs.shape[1] if args.n_channels is None else args.n_channels
    )

    # Plot DDPM sample images
    model = DDPM(args, dim_mults=(1, 2, 4)).to(device)
    if args.trained_model is not None:
        print("Loading pre-trained model!")
        trained_model = torch.load(args.trained_model)["model"]
        model.load_state_dict(trained_model)
        model.eval()
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    count = 0
    gt_image_captions = {}
    tti_image_captions = {}

    eval_dir = os.path.join(os.getcwd(), "model128", "eval_data")
    gt_dir = os.path.join(eval_dir, "gt")
    gt_images_dir = os.path.join(gt_dir, "images")
    tti_dir = os.path.join(eval_dir, "tti")
    tti_images_dir = os.path.join(tti_dir, "images")
    if not os.path.exists(gt_images_dir):
        os.makedirs(gt_images_dir)
    if not os.path.exists(tti_images_dir):
        os.makedirs(tti_images_dir)

    # import pdb;pdb.set_trace()

    for i, data in enumerate(val_loader):
        for j in range(data[0].shape[0]):
            image = reverse_transform(data[0][j])
            caption = data[1][j]
            encoded_caption = tokenize_labels(caption, tokenizer=tokenizer).to(device)
            generated_images = model.generate(num_samples=1, captions=encoded_caption)
            generated_image = reverse_transform(generated_images[-1])
            
            gt_image_name = f"gt_image{count}"
            tti_image_name = f"tti_image{count}"           

            gt_image_captions[gt_image_name] = caption
            tti_image_captions[tti_image_name] = caption

            image.save(os.path.join(gt_images_dir, gt_image_name+".jpg"))
            generated_image.save(os.path.join(tti_images_dir, tti_image_name+".jpg"))
            
            count += 1

        gt_json = json.dumps(gt_image_captions, indent=4)
        tti_json = json.dumps(tti_image_captions, indent=4)
        with open(os.path.join(gt_dir, "gt_captions.json"), "w") as outfile:
            outfile.write(gt_json)
        with open(os.path.join(tti_dir, "tti_captions.json"), "w") as outfile:
            outfile.write(tti_json)



if __name__=="__main__":
    main()

# EVALUATION INSTRUCTIONS
# Run this script to generate images (might want to generate stats for a subset of all images, currently I do this manually by making copies of the images folder w images, e.g. images_1024, and manually making a copy of the json captions file to match this number of images, e.g. tti_captions_1024.json)
# To obtain CLIPScore:
#     - Install clipscore: https://github.com/jmhessel/clipscore (Note, need to install dependencies first: OpenAI CLIP and PyCOCOEvalCap)
#     - Run the following commands to obtain CLIPScore of gt and tti: 
#         - python clipscore.py ~/alignDRAW/eval_data/gt/gt_captions.json ~/alignDRAW/eval_data/gt/images_1024/
#         - python clipscore.py ~/alignDRAW/eval_data/tti/tti_captions_1024.json ~/alignDRAW/eval_data/tti/images_1024/
# To obtain FID Score:
#     - Install pytorch-fid: https://github.com/mseitzer/pytorch-fid
#     - Run the following command:
#         - python -m pytorch_fid ~/alignDRAW/eval_data/gt/images_1024/ ~/alignDRAW/eval_data/tti/images_1024/