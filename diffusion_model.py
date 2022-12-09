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
            noise = torch.rand_like(img)

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
    def generate(self, num_samples, captions):
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
            # The image dimesnion is B x A (According to the DRAW paper).
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

    # @torch.no_grad
    # def generate(self, num_samples):
    #     return self.p_sample_loop(shape=(num_samples, self.channels, self.iamge_size, self.iamge_size))


class DRAWModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.T = args.T
        self.A = args.input_image_size
        self.B = args.input_image_size
        self.z_size = args.z_size
        self.read_N = args.read_N
        self.write_N = args.write_N
        self.enc_size = args.enc_size
        self.dec_size = args.dec_size
        self.device = device
        self.channel = args.n_channels

        # Stores the generated image for each time step.
        self.cs = [0] * self.T

        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        self.encoder = nn.LSTMCell(
            2 * self.read_N * self.read_N * self.channel + self.dec_size, self.enc_size
        )

        # To get the mean and standard deviation for the distribution of z.
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

        self.fc_write = nn.Linear(
            self.dec_size, self.write_N * self.write_N * self.channel
        )

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.dec_size, 5)

    def forward(self, x):
        self.batch_size = x.size(0)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device
        )
        h_dec_prev = torch.zeros(
            self.batch_size, self.dec_size, requires_grad=True, device=self.device
        )

        enc_state = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device
        )
        dec_state = torch.zeros(
            self.batch_size, self.dec_size, requires_grad=True, device=self.device
        )

        for t in range(self.T):
            c_prev = (
                torch.zeros(
                    self.batch_size,
                    self.B * self.A * self.channel,
                    requires_grad=True,
                    device=self.device,
                )
                if t == 0
                else self.cs[t - 1]
            )
            # Equation 3.
            x_hat = x - torch.sigmoid(c_prev)
            # Equation 4.
            # Get the N x N glimpse.
            r_t = self.read(x, x_hat, h_dec_prev)
            # Equation 5.
            h_enc, enc_state = self.encoder(
                torch.cat((r_t, h_dec_prev), dim=1), (h_enc_prev, enc_state)
            )
            # Equation 6.
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc)
            # Equation 7.
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # Equation 8.
            self.cs[t] = c_prev + self.write(h_dec)

            h_enc_prev = h_enc
            h_dec_prev = h_dec

    def read(self, x, x_hat, h_dec_prev):
        # Using attention
        (Fx, Fy), gamma = self.attn_window(h_dec_prev, self.read_N)

        def filter_img(img, Fx, Fy, gamma):
            Fxt = Fx.transpose(self.channel, 2)
            if self.channel == 3:
                img = img.view(-1, 3, self.B, self.A)
            elif self.channel == 1:
                img = img.view(-1, self.B, self.A)

            # Equation 27.
            glimpse = torch.matmul(Fy, torch.matmul(img, Fxt))
            glimpse = glimpse.view(-1, self.read_N * self.read_N * self.channel)

            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)

        return torch.cat((x, x_hat), dim=1)
        # No attention
        # return torch.cat((x, x_hat), dim=1)

    def write(self, h_dec):
        # Using attention
        # Equation 28.
        w = self.fc_write(h_dec)
        if self.channel == 3:
            w = w.view(self.batch_size, 3, self.write_N, self.write_N)
        elif self.channel == 1:
            w = w.view(self.batch_size, self.write_N, self.write_N)

        (Fx, Fy), gamma = self.attn_window(h_dec, self.write_N)
        Fyt = Fy.transpose(self.channel, 2)

        # Equation 29.
        wr = torch.matmul(Fyt, torch.matmul(w, Fx))
        wr = wr.view(self.batch_size, self.B * self.A * self.channel)

        return wr / gamma.view(-1, 1).expand_as(wr)
        # No attention
        # return self.fc_write(h_dec)

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)

        # Equation 1.
        mu = self.fc_mu(h_enc)
        # Equation 2.
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def attn_window(self, h_dec, N):
        # Equation 21.
        params = self.fc_attention(h_dec)
        gx_, gy_, log_sigma_2, log_delta_, log_gamma = params.split(1, 1)

        # Equation 22.
        gx = (self.A + 1) / 2 * (gx_ + 1)
        # Equation 23
        gy = (self.B + 1) / 2 * (gy_ + 1)
        # Equation 24.
        delta = (max(self.A, self.B) - 1) / (N - 1) * torch.exp(log_delta_)
        sigma_2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma_2, delta, N), gamma

    def filterbank(self, gx, gy, sigma_2, delta, N, epsilon=1e-8):
        grid_i = torch.arange(
            start=0.0, end=N, device=self.device, requires_grad=True,
        ).view(1, -1)

        # Equation 19.
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta
        # Equation 20.
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta

        a = torch.arange(0.0, self.A, device=self.device, requires_grad=True).view(
            1, 1, -1
        )
        b = torch.arange(0.0, self.B, device=self.device, requires_grad=True).view(
            1, 1, -1
        )

        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        sigma_2 = sigma_2.view(-1, 1, 1)

        # Equations 25 and 26.
        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * sigma_2))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * sigma_2))

        Fx = Fx / (Fx.sum(2, True).expand_as(Fx) + epsilon)
        Fy = Fy / (Fy.sum(2, True).expand_as(Fy) + epsilon)

        if self.channel == 3:
            Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
            Fx = Fx.repeat(1, 3, 1, 1)

            Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
            Fy = Fy.repeat(1, 3, 1, 1)

        return Fx, Fy

    def loss(self, x):
        self.forward(x)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        # Reconstruction loss.
        # Only want to average across the mini-batch, hence, multiply by the image dimensions.
        Lx = criterion(x_recon, x) * self.A * self.B * self.channel
        # Latent loss.
        Lz = 0

        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_loss = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - 0.5 * self.T
            Lz += kl_loss

        Lz = torch.mean(Lz)
        net_loss = Lx + Lz

        return net_loss

    def generate(self, num_output):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size, device=self.device)

        for t in range(self.T):
            c_prev = (
                torch.zeros(
                    self.batch_size, self.B * self.A * self.channel, device=self.device
                )
                if t == 0
                else self.cs[t - 1]
            )
            z = torch.randn(self.batch_size, self.z_size, device=self.device)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

        imgs = []

        for img in self.cs:
            # The image dimesnion is B x A (According to the DRAW paper).
            img = img.view(-1, self.channel, self.B, self.A)
            imgs.append(
                vutils.make_grid(
                    torch.sigmoid(img).detach().cpu(),
                    nrow=int(np.sqrt(int(num_output))),
                    padding=1,
                    normalize=True,
                    pad_value=1,
                )
            )

        return imgs
