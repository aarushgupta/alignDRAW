import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Directory containing the data.

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
        root = "/data/datasets/coco/"

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
            dset.CocoCaptions(
                f"{root}/train2014",
                annFile=f"{root}/annotations/captions_train2014.json",
                # transform=transforms.Compose([transforms.ToTensor()]),
                transform=img_transform,
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )

        val_dataloader = torch.utils.data.DataLoader(
            dset.CocoCaptions(
                f"{root}/val2014",
                annFile=f"{root}/annotations/captions_val2014.json",
                # transform=transforms.Compose([transforms.ToTensor()]),
                transform=img_transform,
            ),
            batch_size=args.batch_size,
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

    else:
        raise NotImplementedError

    return [train_dataloader, val_dataloader]
