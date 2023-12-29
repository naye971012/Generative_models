import torch
import torch.nn as nn
import numpy as np
import argparse
import wandb
import os
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################
from common.dataset import get_loaders
from trainer.trainer import train
from model import get_model


def main(args):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    args.device = device
    
    train_lodaer, valid_loader, _ = get_loaders(args)
    
    diffusion, model = get_model(args)

    train(args=args,
          train_loader=train_lodaer,
          valid_loader=valid_loader,
          model=model,
          diffusion=diffusion
          )
    

def get_args():
    
    parser = argparse.ArgumentParser(description='Diffusion basic parser')

    #Data arguments
    parser.add_argument('--data_name', type=str, default="cifar10", \
                                        help='name of training data')
    parser.add_argument('--data_save_path', type=str, default="data", \
                                        help='path data will be saved')
    parser.add_argument('--image_size', type=int, default=32, help='data image size')
    parser.add_argument('--channel_size', type=int, default=3, help='image channel size')

    #model setting
    parser.add_argument('--model_name', type=str, default="DDPM", help='determine model')
    parser.add_argument('--model_dim', type=int, default=32, help='determine model size')
    parser.add_argument('--noise_steps', type=int, default=500, help='noise step')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='noise min')
    parser.add_argument('--beta_end', type=float, default=2e-2, help='noise max')

    #Train arguments
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')


    #Save arguments
    parser.add_argument('--logging', type=bool, default=True, help='log train/validation in wandb')
    parser.add_argument('--save_model', type=bool, default=False, help='save model')
    parser.add_argument('--save_path', type=str, default='model_pth/diffusion/', help='save model path')
    parser.add_argument('--save_step_image', type=bool, default=True, help='save generated image step by step')
    parser.add_argument('--image_save_path', type=str, default='Diffusion/output_image/DDPM', help='image save path')

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__=="__main__":
    args = get_args()
    
    # Initialize wandb with specific settings
    if args.logging:
        wandb.init(
            project='GAN_Diffusion',  # Set your project name
            name=f"DDPM_cifar10_size{args.model_dim}",
            config={                       # Set configuration parameters (optional)
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'epoch': args.epoch,
                'data_name': args.data_name,
                'model_dim': args.model_dim,
                'noise_step': args.noise_steps
            }
        )
    
    
    main(args)