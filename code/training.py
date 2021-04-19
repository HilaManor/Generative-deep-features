import models
import torch.nn as nn
import torch.optim as optim

def train(opt, Gs, Zs, reals, NoiseAmp):
    # Todo: load image
    #   Todo: created multi-scale images
    #

    #   Todo: Create SCALES (while)
    nfc = 32 # / 64? TODO
    min_nfc = 32 # / 64? TODO

    G = init_generator(opt)


    G = train_single_scale(G, opt)



def init_generator(opt):
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    # TODO load from file?
    print(netG)
    return netG



def train_single_scale(G, opt):
    # TODO get image
    # TODO pad Image
    #   TODO create noise (and pad it?)

    # Setup Optimizer
    optimizer = optim.Adam(G.parameters(), lr=0.0005, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000],
                                               gamma=0.1)

    for epoch in range(2000):
        # z_opt = padded noise (by noise) (only in first scale changes every iter)
        # noise_ = padded noise (by noise)
        #
        # for every step in D
        #   train by real image
        #   if first step & first epoch
        #       if first scale
        #           in_s = zeros
        #           prev = padded zeros (by image)
        #           z_prev = padded zeros (by noise)
        #           noise_amp = 1
        #       else
        #           prev = padded draw_concat (by image)
        #           z_prev = draw_concat
        #
        #
        #


        scheduler.step()

    # TODO save network?
    return G