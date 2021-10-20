"""A test script for harmonizing elements pasted in an image, or editing an image (changing existing elements in it)
"""

from config import get_arguments
import torch
import random
import os
import image_helpers
import image_processing
import output_handler
import plotting_helpers
import tests
import json

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--ref_path', help='Image path for harmonization or editing by (a naively edited image)', 
                        required=True)
    parser.add_argument('--mask_path', help='mask image path', required=True)
    parser.add_argument('--harmonization_start_scale', help='harmonization injection scale',
                        type=int, required=True)
    parser.add_argument('--editing', action='store_true',
                        help='specify if to perform color editing rather than harmonization (different mask dilate)')
    opt = parser.parse_args()
    
    # Load the trained model parameters according to the params.txt file in the folder
    opt = output_handler.load_parameters(opt, opt.trained_net_dir)

    basename = os.path.basename(opt.image_path)
    basename = basename[:basename.rfind('.')]

    real_img = image_helpers.read_image(opt.image_path, opt.nc, opt.is_cuda)
    real_resized, scale_factor, total_scales = image_processing.preprocess_image(real_img, opt)
    opt.nzx = real_resized.shape[2]
    opt.nzy = real_resized.shape[0]

    out_dir = os.path.join(opt.trained_net_dir, 'Harmonization')
    os.makedirs(out_dir, exist_ok=True)

    Generators, z_opts, NoiseAmp, reals = output_handler.load_network(opt.trained_net_dir)

    if (opt.harmonization_start_scale < 1) | (opt.harmonization_start_scale > (len(Generators) - 1)):
        raise Exception("injection scale should be between 1 and %d" % (len(Generators) - 1))
    else:
        # read the naively edited image and mask and resize them to the coarsest scale
        ref_img = image_helpers.read_image(opt.ref_path, opt.nc, opt.is_cuda)
        mask = image_helpers.read_image(opt.mask_path, opt.nc, opt.is_cuda)
        if ref_img.shape[3] != real_resized.shape[3]:
            mask = image_processing.resize_to_shape(mask,
                                                    [real_resized.shape[2], real_resized.shape[3]],
                                                    opt.nc, opt.is_cuda)
            mask = mask[:, :, :real_resized.shape[2], :real_resized.shape[3]]
            ref_img = image_processing.resize_to_shape(ref_img, [real_resized.shape[2], real_resized.shape[3]], opt.nc, opt.is_cuda)
            ref_img = ref_img[:, :, :real_resized.shape[2], :real_resized.shape[3]]
        
        r = 20 if opt.editing else 7  # Choose dilation according to the task
        mask = image_processing.dilate_mask(mask, opt.is_cuda, radius=r)

        N = len(reals) - 1
        n = opt.harmonization_start_scale
        # resize the naively edited image to the injection scale (This is done after resizing 
        # to the coarsest scales to keep the dimensions the same)
        in_s = image_processing.resize(ref_img, pow(scale_factor, (N - n + 1)), opt.nc, opt.is_cuda)
        in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
        in_s = image_processing.resize(in_s, 1 / scale_factor, opt.nc, opt.is_cuda)
        in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]

        # generate a sample
        out = tests.generate_random_sample(Generators[n:], z_opts[n:], scale_factor, NoiseAmp[n:],
                                           reals[n:], opt=opt, fake=in_s, n=n)
        out = (1 - mask) * real_resized + mask * out[-1]  # Combine with mask
        plotting_helpers.show_im(out)
        plotting_helpers.save_im(out, out_dir, f'Harmonized_{basename}_at{n}_r{r}', convert=True)
