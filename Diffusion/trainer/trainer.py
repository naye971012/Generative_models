import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.DDPM import *
import wandb
from common import *

def train(args, 
          train_loader, 
          valid_loader,
          model=UNet, 
          diffusion=Diffusion,
          ):
    """
    train model to predict denoise
    """
    device = args.device
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    step=0
    for epoch in range(args.epoch):

        tbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for i, (images, _) in enumerate(tbar):
            images = images.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = criterion(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tbar.set_postfix(MSE=loss.item())
            step+=1
            
            if step==1 and args.save_step_image:
                save_image(args,"train",images)
            
            if step%10==0 and args.logging:
                wandb.log({"MSE":loss.item()})
            
            if step%500==0 and args.save_step_image:
                sampled_images, denoise_arr = diffusion.sample(model, n=images.shape[0])
                save_image(args, step, sampled_images.to("cpu").detach())
                save_image(args, f"denoise_{step}", torch.stack(denoise_arr) , flag=True)
                
        if args.save_model:
            models_dict = {
                'model': model.state_dict()
            }
            torch.save(models_dict, args.save_path)
        
        validate(args, model, diffusion, valid_loader, device)

def validate(args, model, diffusion, valid_lodaer, device):
    
    #TODO - i train model on cpu env, calculating FID score takes too long time
    pass