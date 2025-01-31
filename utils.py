import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torchvision.utils as vutils


# Function to generate new images and save the time-steps as an animation.
def generate_image(args, epoch, model):
    x = model.generate(64)
    fig = plt.figure(figsize=(16, 16))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in x]
    anim = animation.ArtistAnimation(
        fig, ims, interval=500, repeat_delay=1000, blit=True
    )
    anim.save(
        f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/draw_epoch_{epoch}.gif",
        dpi=100,
        writer="imagemagick",
    )
    plt.close("all")


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", default=25, type=int, help="Number of glimpses")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument(
        "--input_image_size", default=32, type=int, help="Model input image side"
    )
    parser.add_argument(
        "--z_size", default=100, type=int, help="Latent space dimensions"
    )
    parser.add_argument(
        "--read_N", default=6, type=int, help="N x N dimension of reading glimpse"
    )
    parser.add_argument(
        "--write_N", default=6, type=int, help="N x N dimension of writing glimpse"
    )
    parser.add_argument(
        "--dec_size", default=400, type=int, help="Hidden dimension for decoder"
    )
    parser.add_argument(
        "--enc_size", default=400, type=int, help="Hidden dimension for encoder"
    )
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--dataset_name", default="coco", type=str)
    parser.add_argument("--n_channels", default=3, type=int)
    parser.add_argument(
        "--save_after",
        default=10,
        type=int,
        help="Interval (in epochs) to save results after",
    )
    parser.add_argument(
        "--save_dir", default="./results", type=str, help="Directory to save results to"
    )
    parser.add_argument("--run_idx", default=0, type=int)
    return parser.parse_args()


def plot_sample_images(args, train_loader, device, dataset_name):
    # Plot the training images.
    sample_batch = next(iter(train_loader))
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                sample_batch[0].to(device)[:64],
                nrow=8,
                padding=1,
                normalize=True,
                pad_value=1,
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig(
        f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/training_data_{dataset_name}"
    )
    return


def plot_training_losses(args, losses, dataset_name):
    # Plot the training losses.
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.savefig(
        f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/loss_curve_{dataset_name}"
    )

