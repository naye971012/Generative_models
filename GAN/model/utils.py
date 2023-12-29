
def get_model(args):
    
    if (args.model_name=='DCGAN'):
        from model.DCGAN import Discriminator, Generator, weights_init
        G = Generator(args.img_channel,
                      args.latent_size,
                      args.model_channel_size).apply(weights_init)
        D = Discriminator(args.img_channel,
                          args.model_channel_size).apply(weights_init)
        
    else:
        raise "no model name found"
    
    return G, D