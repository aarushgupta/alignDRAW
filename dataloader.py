import os
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from datasets import MNIST_Captions, COCO_Captions
from utils import download_mnist_captions_dataset, download_coco_processed_dataset

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import numpy as np

from utils import (
    generate_image,
    get_train_parser,
    plot_sample_images,
    plot_training_losses,
    get_validation_loss,
)


"""
MS-COCO dataset is to be downlaoded/setup in the following directory structure

root
  |
  |--annotations
  |--train2014
  |--val2014
  |--test2014

"""


def tokenize_labels(input):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    if isinstance(input, list):
        chosen_caption_idx = np.random.randint(0, len(input))
        chose_captions = input[chosen_caption_idx]
        return tokenizer(
            chose_captions, padding="max_length", truncation=True, return_tensors="pt",
        )
    return tokenizer(input, padding="max_length", truncation=True, return_tensors="pt")


def get_data(args):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """

    if args.dataset_name == "coco":
        root = "/data/datasets/COCO"
        # root = "/home/mateo/Data/datasets/COCO/"

        # if not os.path.exists(f"./data/coco_captions/train-images-32x32.npy"):
        #     print(f"Original data not found, downloading...\n")
        #     download_coco_processed_dataset(f"./data/coco_captions")

        # Transforms for MS-COCO dataset
        img_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((args.input_image_size, args.input_image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

        # COCO dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            dset.CocoCaptions(
                f"{root}/train2014",
                annFile=f"{root}/annotations/captions_train2014.json",
                transform=img_transform,
                target_transform=tokenize_labels,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        val_dataloader = torch.utils.data.DataLoader(
            dset.CocoCaptions(
                f"{root}/val2014",
                annFile=f"{root}/annotations/captions_val2014.json",
                transform=img_transform,
                target_transform=tokenize_labels,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    elif args.dataset_name == "mnist":

        root = "data/"
        train_dataloader = torch.utils.data.DataLoader(
            dset.MNIST(
                root="data/",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dset.MNIST(
                root="data/",
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )

    elif args.dataset_name == "mnist_captions":

        if not os.path.exists(f"./data/mnist_authors/mnist.h5"):
            print(f"Original data not found, downloading...\n")
            download_mnist_captions_dataset("./data/mnist_captions/")

        generate = not os.path.exists("./data/mnist_captions/val_images.npy")
        banned = [random.randint(0, 10) for i in range(12)]

        datafile = (
            f"./data/mnist_authors/mnist.h5" if generate else "./data/mnist_captions/"
        )

        train_dataloader = torch.utils.data.DataLoader(
            MNIST_Captions(
                datafile=datafile,
                generate=generate,
                banned=banned,
                split="train",
                transforms=transforms.Compose([transforms.ToTensor()],),
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )

        val_dataloader = torch.utils.data.DataLoader(
            MNIST_Captions(
                datafile=datafile,
                generate=generate,
                banned=banned,
                split="val",
                transforms=transforms.Compose([transforms.ToTensor()],),
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )

    elif args.dataset_name == "fmnist":
        root = "./data/"
        image_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

        train_dataloader = torch.utils.data.DataLoader(
            dset.FashionMNIST(
                root=root, train=True, download=True, transform=image_transform,
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dset.FashionMNIST(
                root=root, train=False, download=True, transform=image_transform,
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )

    else:
        raise NotImplementedError

    return [train_dataloader, val_dataloader]


if __name__ == "__main__":
    args = get_train_parser()
    train_loader, val_loader = get_data(args)
    import pdb

    pdb.set_trace()

    for i, data in enumerate(train_loader):
        print(i)

