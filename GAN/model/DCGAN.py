import torch
import torch.nn as nn
import numpy as np
from typing import *

#refer https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self,
                 img_channel:int=3,
                 latent_size:int=100,
                 model_channel_size:int=16) -> None:
        super().__init__()
        
        self.img_channel = img_channel
        self.latent_size = latent_size
        self.model_c = model_channel_size
        
        self.init_param()
    
    def init_param(self):
        
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d( self.latent_size, self.model_c * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.model_c * 8),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(self.model_c * 8, self.model_c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_c * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( self.model_c * 4, self.model_c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_c * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( self.model_c * 2, self.model_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_c),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( self.model_c, self.img_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. ``(nc) x 64 x 64``
        )
    
    def generate_latent_vector(self, batch_size:int=1):
        """
        return gausian distribution vector
        size= [batch, channel, width, height]
        """
        latent = np.random.normal(loc=0.0, scale=0.2, size=(batch_size,self.latent_size,1,1))
        return torch.tensor(latent, requires_grad=False, dtype=torch.float32)
    
    def sample(self, num_samples:int):
        return self.generate_latent_vector(num_samples)
    
    def forward(self,x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self,
                 img_channel:int=3,
                 model_channel_size:int=16) -> None:
        super().__init__()
        self.img_channel = img_channel
        self.model_c = model_channel_size
        
        self.init_param()
    
    def init_param(self):
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 ``(nc) x 64 x 64`` 입니다
            nn.Conv2d(self.img_channel, self.model_c, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf) x 32 x 32``
            nn.Conv2d(self.model_c, self.model_c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*2) x 16 x 16``
            nn.Conv2d(self.model_c * 2, self.model_c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*4) x 8 x 8``
            nn.Conv2d(self.model_c * 4, self.model_c * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*8) x 4 x 4``
            nn.Conv2d(self.model_c * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            #finally, [batch, 1, 1, 1]
        )

    def forward(self, input:torch.Tensor):
        out = self.main(input)
        return out

#check dimension
if __name__=="__main__":
    netG = Generator()
    netD = Discriminator()
    x = netG.generate_latent_vector()
    out = netG(x)
    dis_out = netD(out)
    print(x.shape)
    print(out.shape)
    print(dis_out.shape)