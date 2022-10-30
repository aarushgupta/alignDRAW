import os
import h5py
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

from create_mnist_captions import create_mnist_captions_dataset


class MNIST_Captions(Dataset):
    def __init__(
        self,
        datafile,
        generate,
        banned,
        split="train",
        transforms=transforms.ToTensor(),
        target_transforms=transforms.Compose(
            [transforms.ToTensor(), lambda x: F.one_hot(x, 23),]
        ),
        image_side=60,
    ):
        if generate:
            data = h5py.File(datafile, "r")

            if split == "train":
                images = data["train"]
                labels = data["train_labels"]

            else:
                images = data["validation"]
                labels = data["validation_labels"]

            input_images, captions, input_counts = create_mnist_captions_dataset(
                images, labels, banned
            )
            os.makedirs(f"./data/mnist_captions/", exist_ok=True)
            input_images = input_images.reshape(-1, 1, image_side, image_side)
            np.save(f"./data/mnist_captions/{split}_images.npy", input_images)
            np.save(f"./data/mnist_captions/{split}_captions.npy", captions)
            np.save(f"./data/mnist_captions/{split}_counts.npy", input_counts)
            print(f"{split} data generated and saved.")

        else:
            input_images = np.load(f"./data/mnist_captions/{split}_images.npy")
            captions = np.load(f"./data/mnist_captions/{split}_captions.npy")
            input_counts = np.load(f"./data/mnist_captions/{split}_counts.npy")
            print(f"{split} data loaded.")

        self.input_images = input_images
        self.captions = captions
        self.input_counts = input_counts

        # self.transforms = transforms
        # self.target_transforms = target_transforms
        self.seq_len = 23

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        sample_image = self.input_images[idx]
        sample_caption = self.captions[idx]

        # TODO: Define transforms

        sample_image = torch.from_numpy(sample_image)
        sample_caption = F.one_hot(
            torch.from_numpy(sample_caption).to(torch.int64), self.seq_len
        ).to(sample_image.dtype)

        # return [self.transforms(sample_image), self.target_transforms(sample_caption)]
        return [sample_image, sample_caption]
