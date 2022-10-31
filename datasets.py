from operator import length_hint
import os
import h5py
import numpy as np
import random
import pickle

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
        self.n_words = 23
        self.seq_len = 12

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        sample_image = self.input_images[idx]
        sample_caption = self.captions[idx]

        # TODO: Define transforms

        sample_image = torch.from_numpy(sample_image)
        sample_caption = F.one_hot(
            torch.from_numpy(sample_caption).to(torch.int64), self.n_words
        ).to(sample_image.dtype)

        return [sample_image, sample_caption, self.seq_len]


class COCO_Captions(Dataset):
    def __init__(
        self,
        rootdir="./data/coco_captions/",
        batch_size=1,
        split="train",
        n_words_total=None,
        transform=transforms.ToTensor(),
        target_transform=transforms.Compose(
            [transforms.ToTensor(), lambda x: F.one_hot(x, 23),]
        ),
        image_side=32,
    ):
        self.images = np.load(f"{rootdir}/{split}-images-{image_side}x{image_side}.npy")
        self.captions = np.load(f"{rootdir}/{split}-captions.npy")

        captions_len = np.load(f"{rootdir}/{split}-captions-len.npy")
        self.captions_len = np.array([x[0] for x in captions_len])

        self.cap2ims = pickle.load(open(f"{rootdir}/{split}-cap2im.pkl", "rb"))
        print(f"{split} data loaded.")

        # self.transforms = transforms
        # self.target_transforms = target_transforms
        self.n_words_total = n_words_total
        self.batch_size = batch_size
        self.image_side = image_side

        self.prepare()

    def prepare(self):
        self.len_unique = np.unique(self.captions_len)

        # TODO: Add code for minlen and maxlen of captions

        # Get caption indices corresponding to each length
        self.len2captions = {
            length: np.where(self.captions_len == length)[0]
            for length in self.len_unique
        }

        # Hack
        self.len2captions = {
            k: v for k, v in self.len2captions.items() if len(v) > self.batch_size
        }
        self.len_unique = list(self.len2captions.keys())

    # def reset(self):
    #     self.len2ind = {}

    #     for length in self.len2captions:
    #         self.len2captions[length] = np.random.permutation(self.len2captions[length])
    #         self.len2ind[length] = 0

    #     self.len_unique_copy = copy.copy(self.len_unique)

    def __len__(self):
        return len(self.len2captions.keys())

    # def next(self):

    #     if self.len_unique_copy == []
    #         return -1, -1, -1

    #     sampled_length_idx = np.random.randint(0, len(self.len_unique_copy)-1)

    #     sampled_length = self.len_unique_copy[sampled_length_idx]

    #     idx = self.len2ind[sampled_length]

    def __getitem__(self, idx):

        caption_length = self.len_unique[idx]

        caption_indices = np.random.choice(
            self.len2captions[caption_length], self.batch_size, replace=False
        )
        sample_captions = self.captions[caption_indices]

        img_ids = [self.cap2ims[caption_index] for caption_index in caption_indices]
        sample_images = self.images[img_ids]
        sample_images = sample_images.reshape(-1, 3, self.image_side, self.image_side)
        # # TODO: Define transforms
        sample_images = torch.from_numpy(sample_images)
        sample_captions = F.one_hot(
            torch.from_numpy(sample_captions).to(torch.int64), self.n_words_total
        ).to(sample_images.dtype)

        # return [self.transforms(sample_image), self.target_transforms(sample_caption)]
        return [sample_images, sample_captions, caption_length]
