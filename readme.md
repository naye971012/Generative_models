## 개인 공부용 GAN/Diffusion 모델들 직접 구현해보기

## Model List
- [DCGAN](#1-dcgan-deep-convolutional-gan2015)
- [DDPM](#2-ddpm-denoising-diffusion-probabilistic-models-2020)
- More Information - [naye971012's WandB Report](https://wandb.ai/naye971012/GAN_Diffusion?workspace=user-naye971012)



## 1. DCGAN (Deep Convolutional GAN,2015) 
- How to run
```bash
    python GAN/main.py --data_name cifar10 --image_size 64 --image_channel 3 --model_name DCGAN --logging
```
- Train Arguments
```yaml
    device: cpu
    data_name: cifar10
    image_channel: 3 #image channel size (rgb=3, gray=1)
    image_size: 64 #original image size
    batch_size: 16 
    lr: 0.0002
    epoch: 3
    latent_size: 100 #latent vector size
    model_channel_size: 48 #determine mode size
```
- Output Example (Left: Real, Right: Generated)

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/naye971012/Generative_models/assets/74105909/2ca49333-e60a-4157-a111-52a23c2470d4" style="width: 45%; margin-right: 5px;">
    <img src="https://github.com/naye971012/Generative_models/assets/74105909/3e4e7860-6171-4176-be4d-7d97803508fa" style="width: 45%; margin-left: 5px;">
</div>

- Training Process

![DCGAN_Cifar10](https://github.com/naye971012/Generative_models/assets/74105909/e295646d-528b-47b2-95ed-798a94482aeb)

</br>
</br>
</br>

## 2. DDPM (Denoising Diffusion Probabilistic Models, 2020) 
- How to run
```bash
    python Diffusion/main.py --data_name cifar10 --image_size 32 --image_channel 3 --model_name DDPM --logging
```
- Train Arguments
```yaml
    device: cpu
    data_name: cifar10
    image_channel: 3 #image channel size (rgb=3, gray=1)
    image_size: 32 #original image size
    batch_size: 16
    lr: 0.0002
    epoch: 10
    model_dim: 32 #determine model size
    noise_steps: 500 #used when prepare_noise_schedule
    beta_start: 1e-4 #used when prepare_noise_schedule
    beta_end: 2e-2 #used when prepare_noise_schedule
```
- Output Example (Left: Real, Right: Generated)

<div style="display: flex; justify-content: center;">
    <img src="" style="width: 45%; margin-right: 5px;">
    <img src="" style="width: 45%; margin-left: 5px;">
</div>

- Training Process

![DDPM_Cifar10]()









