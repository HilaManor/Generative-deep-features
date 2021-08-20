from config import get_arguments
import torch
import random
import os
import image_helpers
import image_processing
import output_handler

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--ref_path', help='reference image name', required=True)
    parser.add_argument('--harmonization_start_scale', help='harmonization injection scale',
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

    out_dir = os.path.join(opt.output_folder, 'Harmonization', basename)
    os.mkdir(out_dir, exist_ok=True)

    Generators, Zs, reals, NoiseAmp = output_handler.load_network(opt.trained_net_dir)

    if (opt.harmonization_start_scale < 1) | (opt.harmonization_start_scale > (len(Generators) - 1)):
        raise Exception("injection scale should be between 1 and %d" % (len(Generators) - 1))
    else:
        ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
        mask = functions.read_image_dir(
            '%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]), opt)
        if ref.shape[3] != real.shape[3]:
            mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
            mask = mask[:, :, :real.shape[2], :real.shape[3]]
            ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
            ref = ref[:, :, :real.shape[2], :real.shape[3]]
        mask = functions.dilate_mask(mask, opt)

        N = len(reals) - 1
        n = opt.harmonization_start_scale
        in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
        in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
        in_s = imresize(in_s, 1 / opt.scale_factor, opt)
        in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
        out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n,
                              num_samples=1)
        out = (1 - mask) * real + mask * out
        plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.harmonization_start_scale),
                   functions.convert_image_np(out.detach()), vmin=0, vmax=1)