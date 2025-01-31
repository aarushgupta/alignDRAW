import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils

from torchvision import datasets, transforms
from draw_model import DRAWModel
from dataloader import get_data
from utils import (
    generate_image,
    get_train_parser,
    plot_sample_images,
    plot_training_losses,
)


def main():

    args = get_train_parser()

    # Dictionary storing network parameters.
    params = {
        "T": args.T,  # Number of glimpses.
        "batch_size": args.batch_size,  # Batch size.
        "A": args.input_image_size,  # COCO Image width
        "B": args.input_image_size,  # MNIST Image height
        "z_size": args.z_size,  # Dimension of latent space.
        "read_N": args.read_N,  # N x N dimension of reading glimpse.
        "write_N": args.write_N,  # N x N dimension of writing glimpse.
        "dec_size": args.dec_size,  # Hidden dimension for decoder.
        "enc_size": args.enc_size,  # Hidden dimension for encoder.
        "epoch_num": args.n_epochs,  # Number of epochs to train for.
        "learning_rate": args.lr,  # Learning rate.
        "beta1": args.beta1,
        "clip": args.clip_grad,
        "save_epoch": args.save_after,  # After how many epochs to save checkpoints and generate test output.
        "channel": args.n_channels,
    }

    # Make logging directory
    logging_dir = f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/"
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    # os.makedirs(f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/", exist_ok=True)

    # Use GPU is available else use CPU.
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Fetch train and validation loaders
    train_loader, val_loader = get_data(args)

    sample_imgs = next(iter(train_loader))[0]
    args.n_channels = (
        sample_imgs.shape[1] if args.n_channels is None else args.n_channels
    )

    plot_sample_images(args, train_loader, device, args.dataset_name)

    # Initialize the model and optimizer.
    model = DRAWModel(args, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # List to hold the losses for each iteration.
    # Used for plotting loss curve.
    losses = []
    iters = 0
    avg_loss = 0

    print("-" * 25)
    print("Starting Training Loop...\n")
    print(
        f"Epochs: {args.n_epochs} \nBatch Size: {args.batch_size} \nLength of Train Data Loader: {len(train_loader)}"
    )
    print("-" * 25)

    start_time = time.time()

    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()

        for i, (data, _) in enumerate(train_loader, 0):
            # Get batch size.
            bs = data.size(0)
            # Flatten the image.
            data = data.view(bs, -1).to(device)
            optimizer.zero_grad()
            # Calculate the loss.
            loss = model.loss(data)
            loss_val = loss.cpu().data.numpy()
            avg_loss += loss_val
            # Calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            # Update parameters.
            optimizer.step()

            # Check progress of training.
            if i != 0 and i % 100 == 0:
                print(
                    f"[{epoch+1}/{args.n_epochs}][{i}/{len(train_loader)}]\tLoss: {(avg_loss / 100):.4f}"
                )
                avg_loss = 0

            losses.append(loss_val)
            iters += 1

        avg_loss = 0
        epoch_time = time.time() - epoch_start_time
        print(f"Time Taken for Epoch {epoch + 1}: {epoch_time:.2f}s")
        # Save checkpoint and generate test output.
        if (epoch + 1) % args.save_after == 0:
            save_dir = f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/checkpoint"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "params": params,
                },
                f"{save_dir}/model_epoch_{epoch+1}.pt",
            )

            with torch.no_grad():
                generate_image(args, epoch + 1, model)

    training_time = time.time() - start_time
    print("-" * 50)
    print(f"Training finished!\nTotal Time for Training: {(training_time / 60):.2f}m")
    print("-" * 50)
    # Save the final trained network paramaters.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "params": params,
        },
        f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/checkpoint/model_final_{epoch}",
    )

    # Generate test output.
    with torch.no_grad():
        generate_image(args, args.n_epochs, model)

    plot_training_losses(args, losses, args.dataset_name)


if __name__ == "__main__":
    main()
