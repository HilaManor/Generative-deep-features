"""A test script for generating images from abstract paintaings
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
import training
import json

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--ref_path', help='reference image path, i.e. the painting to be recreated', 
                        required=True)
    parser.add_argument('--quantization_flag', action='store_true',
                        help='specify if to perform color quantization training. '
                             'Only the injection scale is re-trained')
    parser.add_argument('--paint_start_scale', help='harmonization injection scale',
                        type=int, required=True)
    opt = parser.parse_args()
    
    # Load the trained model parameters according to the params.txt file in the folder
    opt = output_handler.load_parameters(opt, opt.trained_net_dir)

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
        # read the painting and resize it to the coarsest scale
        ref_img = image_helpers.read_image(opt.ref_path, opt.nc, opt.is_cuda)
        if ref_img.shape[3] != real_resized.shape[3]:
            ref_img = image_processing.resize_to_shape(ref_img, [real_resized.shape[2], real_resized.shape[3]], opt.nc, opt.is_cuda)
            ref_img = ref_img[:, :, :real_resized.shape[2], :real_resized.shape[3]]

        N = len(reals) - 1
        n = opt.paint_start_scale
        
        # resize the painting to the injection scale (This is done after resizing  to the coarsest 
        # scales to keep the dimensions the same)
        in_s = image_processing.resize(ref_img, pow(scale_factor, (N - n + 1)), opt.nc, opt.is_cuda)
        in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
        in_s = image_processing.resize(in_s, 1 / scale_factor, opt.nc, opt.is_cuda)
        in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]

        if opt.quantization_flag:
            dir2trained_model = os.path.join(out_dir,f"Trained Net {opt.paint_start_scale}")
            real_s = image_processing.resize(real_resized, pow(scale_factor, (N - n)), opt.nc, opt.is_cuda)
            real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            
            # Quantizise the real image to get a color scheme
            real_quant, centers = image_processing.quant(real_s, opt.device)
            plotting_helpers.save_im(real_quant,out_dir,f"real_quant_{opt.paint_start_scale}.png",convert=True)
            plotting_helpers.save_im(in_s,out_dir,f"in_paint_{opt.paint_start_scale}.png",convert=True)
            # Quantizise the input painting to match the color scheme
            in_s = image_processing.quant2centers(ref_img, centers,opt.device)
            in_s = image_processing.resize(in_s, pow(scale_factor, (N - n)), opt.nc, opt.is_cuda)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            plotting_helpers.save_im(in_s,out_dir,f"in_paint_quant_{opt.paint_start_scale}.png",convert=True)
            # If the injection scale for this model was already trained with quantization, just load the network
            if (os.path.exists(dir2trained_model)):
                Generators, z_opts, NoiseAmp, reals = output_handler.load_network(dir2trained_model)
            else:
                Generators, z_opts, NoiseAmp, reals = training.train_paint(opt, Generators, z_opts, reals, NoiseAmp, centers, opt.paint_start_scale, dir2trained_model, total_scales, scale_factor)

        # generate a sample
        out = tests.generate_random_sample(Generators[n:], z_opts[n:], scale_factor, NoiseAmp[n:],
                                           reals[n:], opt=opt, fake=in_s, n=n)
        plotting_helpers.show_im(out[-1])
        plotting_helpers.save_im(out[-1], out_dir, f'paint_started_at{n}_quant={opt.quantization_flag}', convert=True)
