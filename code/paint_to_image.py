from config import get_arguments
import torch
import random
import os
import image_helpers
import image_processing
import output_handler
import plotting_helpers
import tests


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--ref_path', help='reference image path', required=True)
    parser.add_argument('--quantization_flag', type=bool, default=False,
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
            raise NotImplementedError

        out = tests.generate_random_sample(Generators[n:], z_opts[n:], scale_factor, NoiseAmp[n:],
                                           reals[n:], opt=opt, fake=in_s, n=n)
        plotting_helpers.show_im(out[-1])
        plotting_helpers.save_im(out[-1], out_dir, f'paint2_started{n}', convert=True)
