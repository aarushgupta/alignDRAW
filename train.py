import os
import time
import torch
from tqdm import tqdm

import torch.optim as optim

from draw_model import DRAWModel
from aligndraw_model import AlignDRAWModel
from diffusion_model import DDPM
from dataloader import get_data
from utils import (
    generate_image,
    get_train_parser,
    plot_sample_images,
    plot_training_losses,
    get_validation_loss,
)

from torch.utils.tensorboard import SummaryWriter


def main():

    args = get_train_parser()

    writer = SummaryWriter(f"runs/{args.dataset_name}/{args.run_idx}")

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

    # Initialize the model and optimizer.
    # model = DRAWModel(args, device).to(device)
    # model = (
    #     AlignDRAWModel(args, device).to(device)
    #     if args.model_name == "alignDRAW"
    #     else DRAWModel(args, device).to(device)
    # )

    # Plot DDPM sample images
    model = DDPM(args, dim_mults=(1, 2, 4)).to(device)
    plot_sample_images(args, train_loader, device, args.dataset_name)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Scheduler for MS-COCO
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1683, gamma=0.1)

    if args.model_name == "ddpm":
        assert args.no_clip_grad is True, "[WARNING] Gradient clipping enabled"

    # List to hold the losses for each iteration.
    # Used for plotting loss curve.
    losses = []
    iters = 0
    epoch_loss = 0

    print("-" * 25)
    print("Starting Training Loop...\n")
    print(
        f"Epochs: {args.n_epochs} \nBatch Size: {args.batch_size} \nLength of Train Data Loader: {len(train_loader)}"
    )
    print("-" * 25)

    start_time = time.time()
    global_step = 0

    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()

        with tqdm(train_loader, unit="batch") as tepoch:
            # for i, (imgs, captions, seq_len) in enumerate(train_loader, 0):
            # for imgs, captions, seq_len in tepoch:
            for imgs, captions in tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                if len(imgs.shape) > 4:
                    imgs = imgs.squeeze()
                    captions = captions.squeeze()

                # # Hack: MNIST_Captions somehow returns this as a tensor, instead of a single number like MS-COCO: this line converts it into a single number
                # if isinstance(seq_len, torch.Tensor):
                #     seq_len = seq_len[0]

                # Get batch size.
                bs = imgs.shape[0]

                # Move training data to GPU
                imgs = imgs.to(device)
                captions = captions.to(device)

                t = torch.randint(0, args.T, (bs,), device=device).long()

                # Calculate the loss.
                optimizer.zero_grad()

                # loss, reconst_loss, kl_loss = model.loss(imgs, captions[:, :seq_len])
                loss, reconst_loss, kl_loss = model.loss(imgs, t, None, captions)

                loss_val = loss.item()
                epoch_loss += loss_val

                writer.add_scalar("Loss/train_total", loss_val, global_step)
                # writer.add_scalar(
                #     "Loss/train_reconst", reconst_loss.item(), global_step
                # )
                # writer.add_scalar("Loss/train_kl", kl_loss.item(), global_step)

                # Calculate the gradients.
                loss.backward()
                if not args.no_clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_val
                    )

                # Update parameters.
                optimizer.step()
                global_step += 1

                tepoch.set_postfix(loss=loss_val)

                losses.append(loss_val)
                iters += 1

        # scheduler.step()
        epoch_time = time.time() - epoch_start_time
        print(f"Time Taken for Epoch {epoch + 1}: {epoch_time:.2f}s")
        print(f"Epoch {epoch + 1} loss: {epoch_loss / iters}")
        epoch_loss = 0
        iters = 0

        # Save checkpoint, generate test output and calculate validation loss.
        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # "scheduler": scheduler.state_dict(),
                    "args": args.__dict__,
                    "global_step": global_step,
                },
                f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/checkpoint/model_epoch_{epoch+1}",
            )

            with torch.no_grad():
                # captions, seq_len = next(iter(val_loader))[1:]
                captions = next(iter(val_loader))[1]
                captions = captions.squeeze() if len(captions.shape) > 3 else captions
                val_img = generate_image(
                    # args, epoch + 1, model, captions[:, : seq_len.item()]
                    args,
                    epoch + 1,
                    model,
                    captions,
                )
                writer.add_image("Image/validation", val_img, global_step)

            with torch.no_grad():
                val_loss, reconst_loss, kl_loss = get_validation_loss(
                    model, val_loader, device
                )
                writer.add_scalar("Loss/val_total", val_loss, global_step)
                # writer.add_scalar("Loss/val_reconst", reconst_loss, global_step)
                # writer.add_scalar("Loss/val_kl", kl_loss, global_step)

    training_time = time.time() - start_time
    print("-" * 50)
    print(f"Training finished!\nTotal Time for Training: {(training_time / 60):.2f}m")
    print("-" * 50)
    # Save the final trained network paramaters.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict(),
            "args": args.__dict__,
            "global_step": global_step,
        },
        f"{args.save_dir}/{args.dataset_name}/{args.run_idx}/checkpoint/model_final_{epoch}",
    )

    # Generate test output.
    with torch.no_grad():
        # captions, seq_len = next(iter(val_loader))[1:]
        captions = next(iter(val_loader))[1]

        captions = captions.squeeze() if len(captions.shape) > 3 else captions
        # _ = generate_image(args, args.n_epochs, model, captions[:, : seq_len.item()])
        _ = generate_image(args, args.n_epochs, model, captions)

    plot_training_losses(args, losses, args.dataset_name)


if __name__ == "__main__":
    main()
