import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
import torch.nn.functional as F

from draw_model import DRAWModel
from aligndraw_model import AlignDRAWModel
from dataloader import get_data
from utils import (
    generate_image,
    get_train_parser,
    plot_sample_images,
    plot_training_losses,
)


def main():

    args = get_train_parser()

    # Make logging directory
    os.makedirs(
        f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/checkpoint", exist_ok=True
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

    plot_sample_images(args, train_loader, device, args.dataset_name)

    # Initialize the model and optimizer.
    # model = DRAWModel(args, device).to(device)
    model = AlignDRAWModel(args, device).to(device)
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

        for i, (imgs, captions, seq_len) in enumerate(train_loader, 0):
            if len(imgs.shape) > 4:
                imgs = imgs.squeeze()
                captions = captions.squeeze()

            # Get batch size.
            bs = imgs.shape[0]

            # Move training data to GPU
            imgs = imgs.view(bs, -1).to(device)
            captions = captions.to(device)

            # Calculate the loss.
            optimizer.zero_grad()
            loss = model.loss(imgs, captions[:, :seq_len])
            loss_val = loss.cpu().data.numpy()
            avg_loss += loss_val

            # Calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            # Update parameters.
            optimizer.step()

            # Check progress of training.
            if i != 0 and i % args.print_after == 0:
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # "params": params,
                    "args": args.__dict__,
                },
                f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/checkpoint/model_epoch_{epoch+1}",
            )

            with torch.no_grad():
                captions, seq_len = next(iter(val_loader))[1:]
                captions = captions.squeeze() if len(captions.shape) > 3 else captions
                generate_image(args, epoch + 1, model, captions[:, : seq_len.item()])

    training_time = time.time() - start_time
    print("-" * 50)
    print(f"Training finished!\nTotal Time for Training: {(training_time / 60):.2f}m")
    print("-" * 50)
    # Save the final trained network paramaters.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "params": params,
            "args": args.__dict__,
        },
        f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/checkpoint/model_final_{epoch}",
    )

    # Generate test output.
    with torch.no_grad():
        captions, seq_len = next(iter(val_loader))[1:]
        captions = captions.squeeze() if len(captions.shape) > 3 else captions
        generate_image(args, args.n_epochs, model, captions[:, : seq_len.item()])

    plot_training_losses(args, losses, args.dataset_name)


if __name__ == "__main__":
    main()
