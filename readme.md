## 개인 공부용 GAN/Diffusion 모델들 직접 구현해보기

## 1. DCGAN (Deep Convolutional GAN,2015)
- Train Arguments
```yaml
    device: cpu
    data_name: cifar10
    image_channel: 3
    image_size: 64
    batch_size: 16
    lr: 0.0002
    epoch: 3
    latent_size: 100
    model_channel_size: 48
```
- Output Example (Left: Real, Right: Generated)

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/naye971012/Generative_models/assets/74105909/2ca49333-e60a-4157-a111-52a23c2470d4" style="width: 45%; margin-right: 5px;">
    <img src="https://github.com/naye971012/Generative_models/assets/74105909/3e4e7860-6171-4176-be4d-7d97803508fa" style="width: 45%; margin-left: 5px;">
</div>

- Training Process

![DCGAN_Cifar10](https://github.com/naye971012/Generative_models/assets/74105909/e295646d-528b-47b2-95ed-798a94482aeb)

- More Information - [WandB Report](https://wandb.ai/naye971012/GAN_Diffusion?workspace=user-naye971012)