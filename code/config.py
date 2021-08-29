import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    gen_group = parser.add_argument_group('General Configuration',
                                          'Define the general running configuration of the project')
    gen_group.add_argument('--image_path', required=True,
                           help='Input image path. Relevant output will be created inside '
                                '<output_folder>/<image basename>')
    gen_group.add_argument('--no_cuda', dest='is_cuda', action='store_false', help='Disable cuda')
    gen_group.add_argument('--output_folder', default='Output', help='Output folder path')
    gen_group.add_argument('--nc', type=int, choices=[1, 3], default=3,
                           help='Number of channels for the images')
    gen_group.add_argument('--manual_seed', type=int, help='Set a manual seed')

    # ~~~~~~~~~~~~~~~~~~ Visualisation Group ~~~~~~~~~~~~~~~~~~~~
    vis_group = parser.add_argument_group('Visualisation Configuration',
                                          'Configure the visualisation parameters')
    vis_group.add_argument('--epoch_print', type=int, default=50,
                               help='Amount of epochs to wait before printing status')
    vis_group.add_argument('--epoch_show', type=int, default=250,
                           help='Amount of epochs to wait before showing mid-image. -1 to not show')
    vis_group.add_argument('--epoch_save', type=int, default=300,
                           help='Amount of epochs to wait before saving mid-image')
    vis_group.add_argument('--generate_fake_amount', type=int, default=5,
                          help='Amount of random samples to be generated')

    # ~~~~~~~~~~~~~~~~~~ Generators HyperParameters Group ~~~~~~~~~~~~~~~~~~~~
    gen_hyper_group = parser.add_argument_group('Generators Hyper-parameters Configuration',
                                            'Set the hyper parameters of the generators')
    gen_hyper_group.add_argument('--ker_size', type=int, default=3,
                                 help='kernel size of the generator')
    gen_hyper_group.add_argument('--num_layer', type=int, default=5,
                        help='number of conv layers in a generator')
    gen_hyper_group.add_argument('--nfc', type=int, default=32,
                                 help='The output  depth of the first layer')
    gen_hyper_group.add_argument('--min_nfc', type=int, default=32,
                                 help='The minimum output depth of any layer')
    gen_hyper_group.add_argument('--padd_size', type=int, default=0, help='Generator pad size')
    gen_hyper_group.add_argument('--pad_type', type=str, default='pre-padding', help='***')
    gen_hyper_group.add_argument('--stride', default=1, help='Generator stride')

    # ~~~~~~~~~~~~~~~~~~ Optimizers HyperParameters Group ~~~~~~~~~~~~~~~~~~~~
    opt_hyper_group = parser.add_argument_group('Optimizers Hyper-parameters Configuration',
                                            'Set the hyper parameters of the optimization process')
    opt_hyper_group.add_argument('--epochs', type=int, default=2000,
                                 help='Number of epochs to train each scale')
    opt_hyper_group.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma, '
                                                                          'multiplying the lr. ')
    opt_hyper_group.add_argument('--lr', type=float, default=0.0005,
                                 help='Learning rate. default=0.0005')
    opt_hyper_group.add_argument('--lr_factor', type=float, default=1.41,  # sqrt(2)
                                 help='Learning rate multiplying factor between scales.'
                                      ' default=1.41')
    opt_hyper_group.add_argument('--beta1', type=float, default=0.5,
                                 help='beta1 for adam. default=0.5')
    opt_hyper_group.add_argument('--Gsteps', type=int, default=1, help='Generator inner steps')  # TODO - doesn't work


    # ~~~~~~~~~~~~~~~~~~ Pyramid Group ~~~~~~~~~~~~~~~~~~~~
    pyrmaid_group = parser.add_argument_group('Pyramid Configuration',
                                              'Configure the generators pyramid architecture')
    pyrmaid_group.add_argument('--min_size', type=int,default=25,
                               help='image minimal size at the coarser scale')
    pyrmaid_group.add_argument('--max_size', type=int,default=250,
                               help='image maximal size at the coarser scale')
    pyrmaid_group.add_argument('--scale_factor', type=float, default=0.75,
                               help='pyramid scale factor')
    pyrmaid_group.add_argument('--noise_amp', type=float, default=0.1,
                               help='addative noise cont weight')
    pyrmaid_group.add_argument('--try_initial_guess', type=str, choices=['true', 'false'],
                                help='Try loading the previous scale\'s weights as an initial guess'
                                     ' if the dimensions match')

    # ~~~~~~~~~~~~~~~~~~  Group ~~~~~~~~~~~~~~~~~~~~
    loss_group = parser.add_argument_group('Loss Function Configuration ',
                                           'Configure the parameters for the loss functions')

    loss_group.add_argument('--loss_func', type=str, default='pdl',
                          help='Loss function for distribution loss, gram/pdl')
    loss_group.add_argument('--alpha', type=float, default=10, help='reconstruction loss weight')
    loss_group.add_argument('--z_opt_zero', action='store_true',
                            help='Sets the reconstruction noise to zero in all scales but the '
                                 'first one')
    loss_group.add_argument('--c_alpha', type=float, default=0,
                          help='The weight of the color loss in respect to other losses')
    loss_group.add_argument('--c_patch_size', type=int, default=5,
                          help='The size of each patch of the color loss calculation')
    loss_group.add_argument('--min_features', type=int,default=100,
                          help='Minimum amount of output features for each vgg-19 layer')
    loss_group.add_argument('--upsample_for_vgg', type=str, choices=['true', 'false'],
                          default='false', help='Upsample the image to minimum size 224 before '
                                                'inserting to VGG instead of adaptivley choosing '
                                                'the layers')
    loss_group.add_argument('--vgg_w1', type=float, default=1,
                            help='The weight of the 1st vgg conv layer in the distribution loss')
    loss_group.add_argument('--vgg_w2', type=float, default=0.75,
                            help='The weight of the 2nd vgg conv layer in the distribution loss')
    loss_group.add_argument('--vgg_w3', type=float, default=0.2,
                            help='The weight of the 3rd vgg conv layer in the distribution loss')
    loss_group.add_argument('--vgg_w4', type=float, default=0.2,
                            help='The weight of the 4th vgg conv layer in the distribution loss')
    loss_group.add_argument('--vgg_w5', type=float, default=0.2,
                            help='The weight of the 5th vgg conv layer in the distribution loss')




    return parser
