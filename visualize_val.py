import os
import time
import torch
from tqdm import tqdm
import imageio
from functools import partial

from diffusion_model import DDPM
from dataloader import get_data, tokenize_labels
from utils import (
    generate_image,
    generate_image_from_caption,
    get_train_parser,
)

from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer
import pickle


def get_sent_from_caption_rep(captions, seq_len):
    x = pickle.load(open("./data/coco_captions/dictionary.pkl", "rb"))

    captions = captions.squeeze().argmax(-1)
    captions = captions[..., :seq_len]
    sentences = []
    ks, vs = list(x.keys()), list(x.values())
    for caption in captions:
        z = [ks[vs.index(word)] for word in caption.numpy()]
        sentences.append(" ".join(z))
    return sentences


def main():

    args = get_train_parser()
    args.n_epochs = None

    # Make logging directory
    os.makedirs(
        f"{args.save_dir}/{args.dataset_name}_{args.model_name}/{args.run_idx}_results",
        exist_ok=True,
    )

    # Use GPU is available else use CPU.
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Fetch train and validation loaders
    train_loader, val_loader = get_data(args)

    sample_imgs = next(iter(train_loader))[0]

    args.n_channels = (
        sample_imgs.shape[1] if args.n_channels is None else args.n_channels
    )

    # Initialize the model and optimizer.
    model = DDPM(args, dim_mults=(1, 2, 4)).to(device)

    # ckpt = torch.load(
    #     f"{args.save_dir}/{args.dataset_name}_{args.model_name}/{args.run_idx}/checkpoint/model_final_{args.n_epochs-1}"
    # )
    ckpt = torch.load(args.trained_model)

    model.load_state_dict(ckpt["model"])

    # Generate test output.
    with torch.no_grad():
        if args.custom_sentence is None:
            captions = next(iter(val_loader))[1]
            captions = (
                {k: v.squeeze().to(device) for k, v in captions.items()}
                if isinstance(captions, dict)
                else captions.to(device)
            )
            _ = generate_image(args, args.n_epochs, model, captions, test=True)
        else:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

            tokenize_roberta = partial(tokenize_labels, tokenizer=tokenizer)

            captions = tokenize_roberta(args.custom_sentence)
            captions = (
                {k: v.to(device) for k, v in captions.items()}
                if isinstance(captions, dict)
                else captions.to(device)
            )
            output = generate_image_from_caption(
                args, args.n_epochs, model, captions, test=True
            )
            imageio.imsave(
                f"{args.save_dir}/{args.dataset_name}_{args.model_name}/{args.run_idx}_results/{args.custom_sentence}.png",
                output.detach().cpu().permute(1, 2, 0).numpy() * 255.0,
            )


if __name__ == "__main__":
    main()
