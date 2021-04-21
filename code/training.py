import models
import functions
import torch.nn as nn
import torch.optim as optim

def train(opt, Gs, Zs, reals, NoiseAmp):
    # TODO load image
    # TODO-FUTURE created multi-scale images

    # TODO-FUTURE Create SCALES (while)
    nfc = 32 # / 64? TODO
    min_nfc = 32 # / 64? TODO

    G = init_generator(opt)


    G, z_curr = train_single_scale(G, opt)



def init_generator(opt):
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    # TODO load from file?
    print(netG)
    return netG



def train_single_scale(Generators, real_imgs, opt):
    real_img = real_imgs[len(Generators)]
    opt.nzx = real_img.shape[2]  # Width of image in current scale
    opt.nzy = real_img.shape[3]  # Height of image in current scale
    # receptive field...
    # the padding amount is determined by the generators amount of layers
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    noise_pad_func = nn.ZeroPad2d(int(pad_noise))
    image_pad_func = nn.ZeroPad2d(int(pad_image))

    # TODO-FUTURE create noise (and pad it?)

    # Setup Optimizer
    optimizer = optim.Adam(G.parameters(), lr=0.0005, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000],
                                               gamma=0.1)

    # TODO arrays for errors for graphs
    style_loss = []

    for epoch in range(opt.epochs):
        # z_opt is {Z*, 0, 0, 0, ...}. The specific set of input noise maps
        # which generates the original image xn
        z_opt = functions.generate_noise([1, opt.nzx, opt.nzy])
        z_opt = noise_pad_func(z_opt.expand(1, 3, opt.nzx, opt.nzy))

        # TODO generate noise z_opt
        # TODO-FUTURE z_opt should only be generated in 1st scale

        # noise_ is the input noise (before adding the image or changing the variance)
        # TODO generate noise noise_
        # TODO-FUTURE pad noises

        # TODO for generic implementation, make
        #       in_s = zeros
        #       prev = padded zeros (by image)
        #       z_prev = padded zeros (by noise)
        #       noise_amp = 1
        # TODO-FUTRE:
        #   first epoch & first step (D)
        #       first scale:
        #           in_s = zeros
        #           prev = padded zeros (by image)
        #           z_prev = padded zeros (by noise)
        #           noise_amp = 1
        #       else:
        #           prev = padded draw_concat 'rand' (by image)
        #               --> prev = previous generated image FROM RANDOM NOISE
        #           z_prev = padded draw_concat 'rec' (by image)
        #               --> z_prev = previous generate image FROM KNOWN NOISE
        #   else (not first epoch || not first step)
        #       prev = padded draw_concat 'rand' (by image)
        #           --> prev = previous generated image FROM (new) RANDOM NOISE

        # TODO noise = noise_
        # TODO-FUTURE if not first scale noise = noise_amp * noise_ + prev

        #  ~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO-THINK for every step in G steps
        #  ~~~~~~~~~~~~~~~~~~~~~~~~

        # TODO zero grad
        # TODO Forward on Generator network
        # TODO Add style_loss: fake + real -> LOSS
        # TODO backward on style_loss

        # TODO-FUTUE implement reconstruction-loss using alpha (We want to ensure that there exists a specific set of input
        #  noise maps, which generates the original image x)
        #      if alpha != 0
        #           Z_opt = noise_amp * z_opt + z_prev
        #               --> in first scale: z_prev=0, z_opt random every iter
        #               --> else: z_prev = previous generate image FROM KNOWN NOISE
        #               -->         z_opt = 0 ({Z*,0,0,0,0,0})
        #           GDS (train) by reconstruction loss
        #               --> .
        #       else:
        #           Z_opt = z_opt (remember, only in first scale the noise changes every ITER)
        #               --> = 0 in all but first scale
        #           rec_loss = 0
        #

        scheduler.step()

    # TODO save network?
    return G, z_opt