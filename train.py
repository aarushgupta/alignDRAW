import os
import time
import torch
from tqdm import tqdm

import torch.optim as optim

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

    writer = SummaryWriter(f"runs/{args.dataset_name}_{args.model_name}/{args.run_idx}")

    # Make logging directory
    os.makedirs(
        f"{args.save_dir}/{args.dataset_name}_{args.model_name}/{args.run_idx}/checkpoint",
        exist_ok=True,
    )

    # Use GPU is available else use CPU.
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Fetch train and validation loaders
    train_loader, val_loader = get_data(args)

    sample_imgs = next(iter(train_loader))[0]

    # DEBUG
    if args.debug:
        from PIL import Image
        import imageio
        import requests

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        imageio.imwrite("./test.png", image)

        from torchvision.transforms import (
            Compose,
            ToTensor,
            Lambda,
            ToPILImage,
            CenterCrop,
            Resize,
        )

        image_size = 128
        transform = Compose(
            [
                Resize(image_size),
                CenterCrop(image_size),
                ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
                Lambda(lambda t: (t * 2) - 1),
            ]
        )

        x_start = transform(image).unsqueeze(0)

        import numpy as np

        reverse_transform = Compose(
            [
                Lambda(lambda t: (t + 1) / 2),
                Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                Lambda(lambda t: t * 255.0),
                Lambda(lambda t: t.numpy().astype(np.uint8)),
                ToPILImage(),
            ]
        )
        image_reversed = reverse_transform(x_start.squeeze())
        imageio.imwrite("./test_1.png", image_reversed)
        exit()

    args.n_channels = (
        sample_imgs.shape[1] if args.n_channels is None else args.n_channels
    )

    # Plot DDPM sample images
    model = DDPM(args, dim_mults=(1, 2, 4)).to(device)
    plot_sample_images(args, train_loader, device, args.dataset_name)

    if args.ft_lang:
        lang_backbone_params = [
            v for k, v in model.named_parameters() if k.startswith("lang_backbone")
        ]
        other_params = [
            v for k, v in model.named_parameters() if not k.startswith("lang_backbone")
        ]
        optimizer = optim.Adam(
            [
                {"params": lang_backbone_params, "lr": args.lr / 1000},
                {"params": other_params, "lr": args.lr},
            ],
            betas=(args.beta1, 0.999),
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
        )

    if args.model_name == "ddpm" or "tti":
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
            for imgs, captions in tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                if len(imgs.shape) > 4:
                    imgs = imgs.squeeze()
                    captions = captions.squeeze()

                bs = imgs.shape[0]

                # Move training data to GPU
                imgs = imgs.to(device)
                captions = (
                    {k: v.squeeze().to(device) for k, v in captions.items()}
                    if isinstance(captions, dict)
                    else captions.to(device)
                )

                t = torch.randint(0, args.T, (bs,), device=device).long()

                optimizer.zero_grad()

                # Calculate the loss.
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
                f"{args.save_dir}/{args.dataset_name}_{args.model_name}/{args.run_idx}/checkpoint/model_epoch_{epoch+1}",
            )

        if (epoch + 1) % args.log_after == 0:
            with torch.no_grad():
                captions = next(iter(val_loader))[1]
                captions = (
                    {k: v.squeeze().to(device) for k, v in captions.items()}
                    if isinstance(captions, dict)
                    else captions.to(device)
                )

                # captions = captions.squeeze() if len(captions.shape) > 3 else captions
                val_img = generate_image(args, epoch + 1, model, captions,)
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
        f"{args.save_dir}/{args.dataset_name}_{args.model_name}/{args.run_idx}/checkpoint/model_final_{epoch}",
    )

    # Generate test output.
    with torch.no_grad():
        captions = next(iter(val_loader))[1]
        captions = (
            {k: v.squeeze().to(device) for k, v in captions.items()}
            if isinstance(captions, dict)
            else captions.to(device)
        )

        _ = generate_image(args, args.n_epochs, model, captions)

    plot_training_losses(args, losses, args.dataset_name)


if __name__ == "__main__":
    main()
