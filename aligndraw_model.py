import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np

"""
The equation numbers on the comments corresponding
to the relevant equation given in the paper:
DRAW: A Recurrent Neural Network For Image Generation.
"""


class AlignDRAWModel(nn.Module):
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
        self.dataset_name = args.dataset_name

        # Lang LSTM params
        self.lang_inp_size = args.lang_inp_size
        self.lang_h_size = args.lang_h_size

        # Align layers params
        self.align_size = args.align_size

        # Stores the generated image for each time step.
        self.cs = [0] * self.T

        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        # Same lists as above for z
        self.logsigmas_z = [0] * self.T
        self.sigmas_z = [0] * self.T
        self.mus_z = [0] * self.T

        self.sent_rep = [0] * self.T

        # Encoder cell
        self.encoder = nn.LSTMCell(
            2 * self.read_N * self.read_N * self.channel + self.dec_size, self.enc_size
        )

        # Q layers (train time)
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        # Z layers (test time)
        self.fc_mu_z = nn.Sequential(nn.Linear(self.enc_size, self.z_size), nn.Tanh())
        self.fc_sigma_z = nn.Sequential(
            nn.Linear(self.enc_size, self.z_size), nn.Tanh()
        )

        # Decoder cell
        self.decoder = nn.LSTMCell(self.z_size + self.lang_h_size * 2, self.dec_size)
        # self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

        # Write layers
        self.fc_write = nn.Linear(
            self.dec_size, self.write_N * self.write_N * self.channel
        )

        # Align layers
        self.fc_align = nn.Sequential(
            nn.Linear(self.lang_h_size * 2 + self.dec_size, self.align_size),
            nn.Tanh(),
            nn.Linear(self.align_size, 1, bias=False),
        )

        # Language LSTM
        self.lang_lstm = nn.LSTM(
            self.lang_inp_size,
            self.lang_h_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.dec_size, 5)

    def forward(self, x, y):
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

        # lang_lstm forward pass
        h_lang, _ = self.lang_lstm(y)  # B x L x 2 * h_out

        for t in range(self.T):
            if t == 0:
                if self.dataset_name == "mnist_captions":
                    c_prev = -10 * torch.ones(
                        self.batch_size,
                        self.B * self.A * self.channel,
                        requires_grad=True,
                        device=self.device,
                    )
                else:
                    c_prev = torch.zeros(
                        self.batch_size,
                        self.B * self.A * self.channel,
                        requires_grad=True,
                        device=self.device,
                    )
            else:
                self.cs[t - 1]

            # Equation 3.
            x_hat = x - torch.sigmoid(c_prev)
            # Equation 4.
            # Get the N x N glimpse.
            # TODO: Might have to process this channel-wise
            r_t = self.read(x, x_hat, h_dec_prev)
            # Equation 5.
            h_enc, enc_state = self.encoder(
                torch.cat((r_t, h_dec_prev), dim=1), (h_enc_prev, enc_state)
            )

            # Equation 6.
            q, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc)
            _, self.mus_z[t], self.logsigmas_z[t], self.sigmas_z[t] = self.sampleZ(
                h_dec_prev
            )

            self.sent_rep[t] = self.align(h_dec_prev, h_lang)

            # Decoder pass
            h_dec, dec_state = self.decoder(
                torch.cat((q, self.sent_rep[t]), dim=-1), (h_dec_prev, dec_state)
            )

            # Equation 8.
            # TODO: Might have to process this channel-wise
            self.cs[t] = c_prev + self.write(h_dec)

            h_enc_prev = h_enc
            h_dec_prev = h_dec
        # (
        #     _,
        #     self.mus_z[self.T],
        #     self.logsigmas_z[self.T],
        #     self.sigmas_z[self.T],
        # ) = self.sampleZ(h_dec_prev)

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

    def align(self, h_dec_prev, h_lang):
        """
        h_dec_prev -> B x h_dec_size
        h_lang -> B x L x 2 * h_lang_size
        """
        align_input = torch.cat(
            (h_dec_prev.unsqueeze(1).repeat(1, h_lang.shape[1], 1), h_lang), dim=-1
        )

        alphas = torch.exp(self.fc_align(align_input))
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)

        return torch.sum(h_lang * alphas, dim=1)

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)

        # Equation 1.
        mu = self.fc_mu(h_enc)
        # Equation 2.
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def sampleZ(self, h_dec_prev):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)

        # Equation 0 from alignDRAW paper
        mu = self.fc_mu_z(h_dec_prev)
        # Equation 2.
        log_sigma = self.fc_sigma_z(h_dec_prev)
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

    def loss(self, x, y):
        self.forward(x, y)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        # Reconstruction loss.
        # Only want to average across the mini-batch, hence, multiply by the image dimensions.
        Lx = criterion(x_recon, x) * self.A * self.B * self.channel
        # Latent loss.
        Lz = 0

        for t in range(self.T):
            mu_q, mu_z = self.mus[t], self.mus_z[t]
            sigma_q, sigma_z = self.sigmas[t], self.sigmas_z[t]
            logsigma_q, logsigma_z = self.logsigmas[t], self.logsigmas_z[t]

            # kl_loss = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - 0.5 * self.T
            kl_loss = (
                0.5
                * torch.sum(
                    (mu_q - mu_z) ** 2
                    + (sigma_q / sigma_z) ** 2
                    - 2 * (logsigma_q - logsigma_z),
                    1,
                )
                - 0.5 * self.T
            )
            Lz += kl_loss

            # Add loss term for

        Lz = torch.mean(Lz)
        net_loss = Lx + Lz

        return net_loss

    def generate(self, num_output, y):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size, device=self.device)
        y = y.to(self.device)

        # lang_lstm pass
        h_lang, _ = self.lang_lstm(y)

        for t in range(self.T):
            c_prev = (
                torch.zeros(
                    self.batch_size, self.B * self.A * self.channel, device=self.device
                )
                if t == 0
                else self.cs[t - 1]
            )

            # Change in alignDRAW, z is sampled from h_dec_prev and h_land rather than independently sampled as in DRAW
            # z = torch.randn(self.batch_size, self.z_size, device=self.device)

            z, _, _, _ = self.sampleZ(h_dec_prev)
            self.sent_rep[t] = self.align(h_dec_prev, h_lang)

            # Decoder pass
            h_dec, dec_state = self.decoder(
                torch.cat((z, self.sent_rep[t]), dim=-1), (h_dec_prev, dec_state)
            )

            # Write operator
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
