import torch
import torch.nn as nn
import numpy as np
import argparse
import wandb
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from common.dataset import get_loaders
from trainer import train,validate
from model import get_model


def main(args):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    train_lodaer, valid_loader, _ = get_loaders(args)
    
    generator, discriminator = get_model(args)

    train(args=args,
          model_D=discriminator,
          model_G=generator,
          train_loader=train_lodaer,
          valid_loader=valid_loader,
          device=device)
    

def get_args():
    
    parser = argparse.ArgumentParser(description='GAN basic parser')

    #Data arguments
    parser.add_argument('--data_name', type=str, default="cifar10", \
                                        help='name of training data')
    parser.add_argument('--data_save_path', type=str, default="data", \
                                        help='path data will be saved')
    parser.add_argument('--image_size', type=int, default=64, help='data image size')
    parser.add_argument('--img_channel', type=int, default=3, help='image channel size')

    parser.add_argument('--model_name', type=str, default="DCGAN", help='determine model')
    parser.add_argument('--model_channel_size', type=int, default=48, help='determine model size')

    #Train arguments
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--latent_size', type=int, default=100, help='latent vector size')

    #Save arguments
    parser.add_argument('--logging', type=bool, default=False, help='log train/validation in wandb')
    parser.add_argument('--save_model', type=bool, default=False, help='save model')
    parser.add_argument('--save_path', type=str, default='model_pth/gan/', help='save model path')
    parser.add_argument('--save_step_image', type=bool, default=True, help='save generated image step by step')
    parser.add_argument('--image_save_path', type=str, default='GAN/output_image/DCGAN', help='image save path')

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__=="__main__":
    args = get_args()
    
    # Initialize wandb with specific settings
    if args.logging:
        wandb.init(
            project='GAN_Diffusion',  # Set your project name
            name="DCGAN_cifar10",
            config={                       # Set configuration parameters (optional)
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'epoch': args.epoch,
                'data_name': args.data_name,
                'model_size': args.model_channel_size
            }
        )
    
    
    main(args)