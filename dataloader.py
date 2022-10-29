import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Directory containing the data.
root = "data/"


def get_data(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose(
        [transforms.Resize(params["A"]), transforms.ToTensor()]
    )

    # MNIST dataloader
    dataloader = torch.utils.data.DataLoader(
        dset.MNIST(
            root,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=params["batch_size"],
        shuffle=False,
    )

    # # Create the dataset.
    # dataset = dset.ImageFolder(root=root, transform=transform)

    # # Create the dataloader.
    # dataloader = torch.utils.data.DataLoader(dataset,
    #     batch_size=params['batch_size'],
    #     shuffle=True)

    return dataloader
