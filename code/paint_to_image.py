from config import get_arguments
import torch
import random
import os
import image_helpers
import image_processing
import output_handler
import plotting_helpers
import tests
import training

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--ref_path', help='reference image path', required=True)
    parser.add_argument('--quantization_flag', action='store_true',
                        help='specify if to perform color quantization training')
    parser.add_argument('--paint_start_scale', help='harmonization injection scale',
                        type=int, required=True)
    opt = parser.parse_args()

    opt.is_cuda = opt.is_cuda and torch.cuda.is_available()
    opt.device = torch.device("cuda:0" if opt.is_cuda else "cpu")
    # preprocess parameters
    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", opt.manual_seed)
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    basename = os.path.basename(opt.image_path)
    basename = basename[:basename.rfind('.')]

    real_img = image_helpers.read_image(opt.image_path, opt.nc, opt.is_cuda)
    real_resized, scale_factor, total_scales = image_processing.preprocess_image(real_img, opt)
    opt.nzx = real_resized.shape[2]
    opt.nzy = real_resized.shape[0]

    out_dir = os.path.join(opt.trained_net_dir, 'Paint2Image')
    os.makedirs(out_dir, exist_ok=True)

    Generators, z_opts, NoiseAmp, reals = output_handler.load_network(opt.trained_net_dir)

    if (opt.paint_start_scale < 1) | (opt.paint_start_scale > (len(Generators) - 1)):
        raise Exception("injection scale should be between 1 and %d" % (len(Generators) - 1))
    else:
        ref_img = image_helpers.read_image(opt.ref_path, opt.nc, opt.is_cuda)
        if ref_img.shape[3] != real_resized.shape[3]:
            ref_img = image_processing.resize_to_shape(ref_img, [real_resized.shape[2], real_resized.shape[3]], opt.nc, opt.is_cuda)
            ref_img = ref_img[:, :, :real_resized.shape[2], :real_resized.shape[3]]

        N = len(reals) - 1
        n = opt.paint_start_scale
        in_s = image_processing.resize(ref_img, pow(scale_factor, (N - n + 1)), opt.nc, opt.is_cuda)
        in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
        in_s = image_processing.resize(in_s, 1 / scale_factor, opt.nc, opt.is_cuda)
        in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]

        if opt.quantization_flag:
            opt.mode = 'paint_train'
            opt.layers_weights = [opt.vgg_w1, opt.vgg_w2, opt.vgg_w3, opt.vgg_w4, opt.vgg_w5]
            opt.chosen_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
            if torch.cuda.is_available() and not opt.is_cuda:
                print("WARNING: You have a CUDA device, consider removing --not_cuda")
            opt.is_cuda = opt.is_cuda and torch.cuda.is_available()
            opt.device = torch.device("cuda:0" if opt.is_cuda else "cpu")
            opt.try_initial_guess = True if opt.try_initial_guess == 'true' else False
            opt.upsample_for_vgg = True if opt.upsample_for_vgg == 'true' else False
            dir2trained_model =  os.path.join(out_dir,f"Trained Net {opt.paint_start_scale}")
            # N = len(reals) - 1
            # n = opt.paint_start_scale
            real_s = image_processing.resize(real_resized, pow(scale_factor, (N - n)), opt.nc, opt.is_cuda)
            real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            real_quant, centers = image_processing.quant(real_s, opt.device)
            plotting_helpers.save_im(real_quant,out_dir,f"real_quant_{opt.paint_start_scale}.png",convert=True)
            plotting_helpers.save_im(in_s,out_dir,f"in_paint_{opt.paint_start_scale}.png",convert=True)
            in_s = image_processing.quant2centers(ref_img, centers,opt.device)
            in_s = image_processing.resize(in_s, pow(scale_factor, (N - n)), opt.nc, opt.is_cuda)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            plotting_helpers.save_im(in_s,out_dir,f"in_paint_quant_{opt.paint_start_scale}.png",convert=True)
            if (os.path.exists(dir2trained_model)):
                # print('Trained model does not exist, training SinGAN for SR')
                Generators, z_opts, NoiseAmp, reals = output_handler.load_network(dir2trained_model)
            else:
                Generators, z_opts, NoiseAmp, reals = training.train_paint(opt, Generators, z_opts, reals, NoiseAmp, centers, opt.paint_start_scale, dir2trained_model, total_scales, scale_factor)

        out = tests.generate_random_sample(Generators[n:], z_opts[n:], scale_factor, NoiseAmp[n:],
                                           reals[n:], opt=opt, fake=in_s, n=n)
        plotting_helpers.show_im(out[-1])
        plotting_helpers.save_im(out[-1], out_dir, f'paint_started_at{n}', convert=True)
