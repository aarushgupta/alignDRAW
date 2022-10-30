import os
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from datasets import MNIST_Captions, COCO_Captions
from utils import download_mnist_captions_dataset, download_coco_processed_dataset


"""
MS-COCO dataset is to be downlaoded/setup in the following directory structure

root
  |
  |--annotations
  |--train2014
  |--val2014
  |--test2014

"""


def get_data(args):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """

    if args.dataset_name == "coco":
        root = "./data/coco_captions"
        # root = "/data/datasets/coco/"

        if not os.path.exists(f"./data/coco_captions/train-images-32x32.npy"):
            print(f"Original data not found, downloading...\n")
            download_coco_processed_dataset(f"./data/coco_captions")

        # Transforms for MS-COCO dataset
        img_transform = transforms.Compose(
            # [transforms.Resize(params["A"]), transforms.ToTensor()]
            [
                transforms.Resize((args.input_image_size, args.input_image_size)),
                transforms.ToTensor(),
            ]
        )

        # COCO dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            COCO_Captions(
                root,
                split="train",
                seq_len=args.lang_inp_size,
                # transform=transforms.Compose([transforms.ToTensor()]),
                transform=img_transform,
                batch_size=args.batch_size,
            ),
            batch_size=1,
            shuffle=False,
        )

        val_dataloader = torch.utils.data.DataLoader(
            COCO_Captions(
                root,
                split="dev",
                seq_len=args.lang_inp_size,
                # transform=transforms.Compose([transforms.ToTensor()]),
                transform=img_transform,
                batch_size=args.batch_size,
            ),
            batch_size=1,
            shuffle=False,
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

    else:
        raise NotImplementedError

    return [train_dataloader, val_dataloader]
