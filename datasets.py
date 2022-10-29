import os
import h5py
import numpy as np
import random

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from create_mnist_captions import create_mnist_captions_dataset


class MNIST_Captions(Dataset):
    def __init__(
        self,
        datafile,
        generate,
        banned,
        split="train",
        transforms=transforms.ToTensor(),
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

        self.transforms = transforms

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        sample_image = self.input_images[idx]
        sample_caption = self.captions[idx]

        # TODO: Define transforms

        return [sample_image, sample_caption]
