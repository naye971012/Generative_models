import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from Diffusion.model.utils import *

class Diffusion:
    def __init__(self, 
                 noise_steps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02, 
                 img_size=(32,32),
                 channel_size=3,
                 device="cuda"):
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w, self.h = img_size
        self.channel_size = channel_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        t: int,
        Args:
            x (torch.tensor): [# of batch, channel, w, h]
            t (int): timestep

        Returns:
            noise images, noise
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        denoise by model and return generated image
        """
        denoising_step_arr = []
        
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.channel_size, self.w, self.h)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps))):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
                                            + torch.sqrt(beta) * noise
                
                if i%10==0:
                    denoising_step_arr.append( ((x[0].clamp(-0.999, 0.999) + 1) / 2).to('cpu').detach() )
        
        model.train()
        x = (x.clamp(-0.999, 0.999) + 1) / 2
        return x, denoising_step_arr


class UNet(nn.Module):
    def __init__(self, 
                 c_in=3, 
                 start_imgsize=64, 
                 model_dim=64, 
                 device="cuda"):
        super().__init__()
        self.device = device
        self.model_dim = model_dim
        
        self.inc = DoubleConv(c_in, model_dim)
        self.down1 = Down(model_dim, model_dim*2 , self.model_dim*4)
        self.sa1 = SelfAttention(model_dim*2, start_imgsize//2)
        self.down2 = Down(model_dim*2, model_dim*4 , self.model_dim*4)
        self.sa2 = SelfAttention(model_dim*4, start_imgsize//4)
        self.down3 = Down(model_dim*4, model_dim*4 , self.model_dim*4)
        self.sa3 = SelfAttention(model_dim*4, start_imgsize//8)

        self.bot1 = DoubleConv(model_dim*4, model_dim*8)
        self.bot2 = DoubleConv(model_dim*8, model_dim*8)
        self.bot3 = DoubleConv(model_dim*8, model_dim*4)

        self.up1 = Up(model_dim*8, model_dim*2 , self.model_dim*4)
        self.sa4 = SelfAttention(model_dim*2, start_imgsize//4)
        self.up2 = Up(model_dim*4, model_dim , self.model_dim*4)
        self.sa5 = SelfAttention(model_dim, start_imgsize//2)
        self.up3 = Up(model_dim*2, model_dim , self.model_dim*4)
        self.sa6 = SelfAttention(model_dim, start_imgsize)
        self.outc = nn.Conv2d(model_dim, c_in, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.model_dim*4)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output