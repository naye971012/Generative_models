import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from torchmetrics.image import PerceptualPathLength
import wandb
import matplotlib.pyplot as plt

REAL_LABEL = 1.
FAKE_LABEL = 0.

def train(args, model_D, model_G, train_loader, valid_loader, device):
    
    #PPL is for validation step
    ppl = PerceptualPathLength(num_samples=10000,
                                     batch_size=args.batch_size,
                                     resize=args.image_size)
    
    optimizerD = optim.Adam(model_D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(model_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    step=0
    for epoch in range(args.epoch):
        epoch_errD = 0.0
        epoch_errG = 0.0
        epoch_D_x = 0.0
        epoch_D_G_z1 = 0.0
        epoch_D_G_z2 = 0.0
        
        tqdm_batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")
        for i, (_image, _label) in enumerate(tqdm_batch):
                
            _image = _image.to(device)
            _label = _label.to(device)
            
            b_size = _label.size(0)
            
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            
            ############################
            # (1) Update Discriminator : maximize log(D(x)) 
            ###########################
            
            model_D.zero_grad()
            
            label = torch.full((b_size,), REAL_LABEL,
                            dtype=torch.float, device=device)
            output = model_D(_image).view(-1)
            
            #train with real label and real image
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()



            #############################
            # (1) Update Discriminator : maximize log(1 - D(G(z)))
            #############################

            #generate vector
            latent_vector = model_G.generate_latent_vector(args.batch_size).to(device)
            
            fake = model_G(latent_vector)
            label.fill_(FAKE_LABEL)
            output = model_D(fake).view(-1)

            #train with fake label and fake image
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            
            #update discriminator
            optimizerD.step()



            ############################
            # (2) Update Generator: maximize log(D(G(z)))
            ###########################
            model_G.zero_grad()
            
            label.fill_(REAL_LABEL)
            fake = model_G(latent_vector)
            output = model_D(fake).view(-1)
            
            #train with real label and fake image
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            #update generator
            optimizerG.step()




            ######### Logging #########
            epoch_D_G_z1+=D_G_z1
            epoch_D_G_z2+=D_G_z2
            epoch_D_x+=D_x
            epoch_errG+=errG.item()
            epoch_errD+=errD.item()
            ###########################
            tqdm_batch.set_postfix(errG=epoch_errG/(i+1),
                                   errD=epoch_errD/(i+1),
                                   Dx=epoch_D_x/(i+1),
                                   DGz1=epoch_D_G_z1/(i+1),
                                   DGz2=epoch_D_G_z2/(i+1))

            if step%30==0 and args.logging:
                wandb.log({"err_Generator": epoch_errG/(i+1),
                       "err_Discriminator": epoch_errD/(i+1),
                       "D_Maximize_D(x)": epoch_D_x/(i+1),
                       "D_Maximize_D(1-G(z))":epoch_D_G_z1/(i+1),
                       "G_Maximize_D(G(z))":epoch_D_G_z2/(i+1)})
            
            
            if step%500==0 and args.save_step_image:
                latent_vector = model_G.generate_latent_vector(args.batch_size).to(device)
                generated_image = model_G(latent_vector)
                save_image(args, step, generated_image.to(device).detach())
                
            step+=1
            
        if args.save_model:
            models_dict = {
                'Generator': model_G.state_dict(),
                'Discriminator': model_D.state_dict()
            }
            torch.save(models_dict, args.save_path)
        
        validate(args, model_G, valid_loader, device, ppl)    
        

def validate(args, 
             model_G, 
             valid_loader, 
             device, 
             ppl:PerceptualPathLength):
    # calculate model score by PPL (perceptual path length score)
    # save generated images in wandb
    
    ppl_mean, ppl_std, ppl_raw = ppl(model_G)
    
    print(f"ppl score: {ppl_mean}, std: {ppl_std}")
    
    if args.logging:
        wandb.log({"ppl_mean": ppl_mean,
                "ppl_std": ppl_std})


def save_image(args, step, image_tensor):
    # 이미지를 subplot에 표시할 수 있도록 설정합니다.
    # Determine the number of rows and columns dynamically based on the batch size
    batch_size = image_tensor.size(0)
    rows = int(batch_size ** 0.5)  # Use the square root of the number of images for rows
    cols = (batch_size + rows - 1) // rows  # Calculate columns based on rows

    # Set up for displaying images in subplots dynamically
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))  # Adjust rows and columns dynamically

    # Place each image into a subplot
    for idx, image in enumerate(image_tensor):
        row = idx // cols  # Calculate row index
        col = idx % cols    # Calculate column index

        # Convert the image tensor to a numpy array for visualization
        image_np = image.permute(1, 2, 0).numpy()  # Rearrange tensor for Matplotlib compatibility

        # Display the image in the subplot
        axes[row, col].imshow( (image_np+1)/2 )
        axes[row, col].axis('off')  # Deactivate axis for the subplot

    # Hide any extra subplots (if the number of images is fewer than rows*columns)
    for i in range(batch_size, rows * cols):
        axes.flatten()[i].axis('off')

    plt.tight_layout()  # Adjust subplot spacing for a better layout
    
    path = f'{args.image_save_path}/step_{step}.png'
    plt.savefig(path)

    if args.logging:
        wandb.log({"generated_images": wandb.Image(path) })