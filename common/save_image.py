import wandb
import matplotlib.pyplot as plt

def save_image(args, step, image_tensor, flag=False):
    """
    Args:
        args (_type_): arguments
        step (int): step of image tensor
        image_tensor (tensor): generated image
        flag (bool, optional): distinguish logging in wandb. Defaults to False.
    """
    
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
        if flag:
            wandb.log({"denoising_images": wandb.Image(path) })
        else:
            wandb.log({"generated_images": wandb.Image(path) })