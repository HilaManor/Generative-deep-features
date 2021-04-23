import models
import functions
import torch
import torch.nn as nn
import torch.optim as optim

def train(opt, Gs, Zs, reals, NoiseAmp):
    # TODO load image
    # TODO-FUTURE created multi-scale images

    # TODO-FUTURE Create SCALES (while)
    nfc = 32 # / 64? TODO
    min_nfc = 32 # / 64? TODO

    curr_G = init_generator(opt)

    G, z_curr = train_single_scale(Gs, curr_G, real_imgs, opt)



def init_generator(opt):
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    # TODO load from file?
    print(netG)
    return netG



def train_single_scale(Generators, curr_G, real_imgs, opt):
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
    optimizer = optim.Adam(curr_G.parameters(), lr=0.0005, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000],
                                               gamma=0.1)

    # TODO arrays for errors for graphs
    style_loss = []

    # TODO-FUTURE currently only for generic implementation (1 scale)
    prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
    # TODO in_s = prev
    prev = image_pad_func(prev)
    z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
    z_prev = noise_pad_func(z_prev)
    opt.noise_amp = 1

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

    for epoch in range(opt.epochs):
        # z_opt is {Z*, 0, 0, 0, ...}. The specific set of input noise maps
        # which generates the original image xn
        z_opt = functions.generate_noise([1, opt.nzx, opt.nzy])
        z_opt = noise_pad_func(z_opt.expand(1, opt.nc_z, opt.nzx, opt.nzy))

        # noise_ is the input noise (before adding the image or changing the variance)
        noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
        noise_ = noise_pad_func(noise_.expand(1, opt.nc_z, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same
        # TODO-FUTURE z_opt should only be generated in 1st scale

        noise = noise_
        # TODO-FUTURE if not first scale noise = noise_amp * noise_ + prev


        # TODO-THINK for every step in G steps
        for j in range(opt.Gsteps):
            curr_G.zero_grad()
            fake_im = curr_G(noise.detach(), prev)  # TODO think on detach

            # TODO Add style_loss: fake + real -> LOSS

            loss.backward()  # TODO retain_graph=True

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

            optimizer.step()
        scheduler.step()

        # update prev
        prev = draw_concat(Generators, 'rand', noise_pad_func, image_pad_func, opt)
        prev = image_pad_func(prev)

    # TODO save network?
    return curr_G, z_opt


def draw_concat(Generators, mode, noise_pad_func, image_pad_func, opt):
    # TODO-FUTURE good luck
    return torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
