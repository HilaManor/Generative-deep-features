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

    # ~~~~~~~~~~~~~~~~~~ Generators HyperParameters Group ~~~~~~~~~~~~~~~~~~~~
    gen_hyper_group = parser.add_argument_group('Generators Hyper-parameters Configuration',
                                            'Set the hyper parameters of the generators')
    gen_hyper_group.add_argument('--ker_size', type=int, default=3, help='kernel size')
    gen_hyper_group.add_argument('--num_layer', type=int, default=5,
                        help='number of conv layers in a generator')
    gen_hyper_group.add_argument('--nfc', type=int, default=32,
                                 help='The output  depth of the first layer')
    gen_hyper_group.add_argument('--min_nfc', type=int, default=32,
                                 help='The minimum output depth of any layer')
    gen_hyper_group.add_argument('--padd_size', type=int, default=0, help='Net pad size')
    gen_hyper_group.add_argument('--stride', default=1, help='stride')

    # ~~~~~~~~~~~~~~~~~~ Optimizers HyperParameters Group ~~~~~~~~~~~~~~~~~~~~
    opt_hyper_group = parser.add_argument_group('Optimizers Hyper-parameters Configuration',
                                                'Set the hyper parameters of the optimization process')
    opt_hyper_group.add_argument('--epochs', type=int, default=2000,
                                 help='Number of epochs to train each scale')
    opt_hyper_group.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma')
    opt_hyper_group.add_argument('--lr', type=float, default=0.0005, help='Learning rate. default=0.0005')
    opt_hyper_group.add_argument('--lr_factor', type=float, default=1.41,  # sqrt(2)
                                 help='Learning rate factor between scales. default=1.41')
    opt_hyper_group.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    opt_hyper_group.add_argument('--Gsteps', type=int, default=1, help='Generator inner steps')  # TODO 3
    opt_hyper_group.add_argument('--alpha', type=float, default=10, help='reconstruction loss weight')

    # ~~~~~~~~~~~~~~~~~~ Pyramid Group ~~~~~~~~~~~~~~~~~~~~
    pyrmaid_group = parser.add_argument_group('Pyramid Configuration',
                                              'Configure the generators pyramid architecture')
    pyrmaid_group.add_argument('--min_size', type=int, help='image minimal size at the coarser scale',
                        default=25)
    pyrmaid_group.add_argument('--max_size', type=int, help='image maximal size at the coarser scale',
                        default=250)
    pyrmaid_group.add_argument('--scale_factor', type=float, default=0.75, help='pyramid scale factor')
    pyrmaid_group.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')

    # ~~~~ TODO
    oo_group = parser.add_argument_group('Ah ', 'ooh')
    oo_group.add_argument('--loss_func', type=str, help='style/pdl', default='pdl')
    oo_group.add_argument('--min_features', type=int, help='aa', default=100)
    oo_group.add_argument('--try_initial_guess', action='store_true',
                          help='Try loading the previous scale\'s weights as an initial guess if '
                               'the dimensions match')
    oo_group.add_argument('--upsample_for_vgg', action='store_true',
                          help='Upsample the image to 224 before inserting to VGG instead of '
                               'addaptivley choosing the layers')
    oo_group.add_argument('--c_patch_size', type=int, default=5, help='aa')
    oo_group.add_argument('--c_alpha', type=float, help='aa', default=0)
    oo_group.add_argument('--generate_fake_amount', type=int, help='aa', default=5)
    oo_group.add_argument('--z_opt_zero', action='store_true', help='aa')

    return parser
