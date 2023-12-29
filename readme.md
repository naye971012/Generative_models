## 개인 공부용 GAN/Diffusion 모델들 직접 구현해보기

## 1. DCGAN (Deep Convolutional GAN,2015)
- Train Arguments
```yaml
    data_name: cifar10
    image_channel: 3
    image_size: 64
    batch_size: 16
    lr: 0.0002
    epoch: 3
    latent_size: 100
    model_channel_size: 48
```
- Output Example

- Training Process
![DCGAN_Cifar10]()

- [WandB Report](https://wandb.ai/naye971012/GAN_Diffusion?workspace=user-naye971012)