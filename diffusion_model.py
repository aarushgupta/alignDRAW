import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils

from diffusion_utils import *


class DDPM(nn.Module):
    def __init__(
        self,
        args,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ) -> None:
        super().__init__()

        self.image_size = args.input_image_size
        dim = args.input_image_size
        self.channels = 3 if args.n_channels is None else args.n_channels

        self.loss_type = "l1"
        self.timesteps = args.T
        self.schedule_type = "linear"

        self.setup_diffusion_process()

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(self.channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                PositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # Main layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, self.channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # Downsample layers
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck layers
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

    def setup_diffusion_process(self):
        if self.schedule_type == "linear":
            self.betas = linear_beta_schedule(self.timesteps)
        else:
            raise NotImplementedError

        # Define alphas
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Diffusion calculations q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        # Posterior calculations q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )

    def sample_q(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def loss(self, img, t, noise=None, captions=None):
        if noise is None:
            noise = torch.randn_like(img)

        x_noisy = self.sample_q(x_start=img, t=t, noise=noise)
        predicted_noise = self.forward(x_noisy, t)

        if self.loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError

        return loss, None, None

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in paper
        # Use the model (noise predictor) to predict the mean (?)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.forward(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            # Algorithm 2 line 4
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    # def p_sample_loop(self, num_samples):
    def generate(self, num_samples, captions=None):
        shape = (num_samples, self.channels, self.image_size, self.image_size)
        device = next(self.parameters()).device

        image = torch.randn(shape, device=device)
        denoised_images = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="Sampling loop time step",
            total=self.timesteps,
        ):
            image = self.p_sample(
                image, torch.full((num_samples,), i, device=device, dtype=torch.long), i
            )
            denoised_images.append(image)
        # return denoised_images

        img_grid = []

        for img in denoised_images:
            img = img.view(-1, self.channels, self.image_size, self.image_size)
            img_grid.append(
                vutils.make_grid(
                    # torch.sigmoid(img).detach().cpu(),
                    ((img + 1) / 2).detach().cpu(),
                    nrow=int(np.sqrt(int(num_samples))),
                    padding=1,
                    normalize=True,
                    pad_value=1,
                )
            )

        return img_grid
