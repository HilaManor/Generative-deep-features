"""The main file that includes training the model and generating random samples at the end"""

from config import get_arguments
import torch
import random
import os
import image_processing, image_helpers
import training
from output_handler import gen_unique_out_dir_path, save_network
from tests import run_tests
import json
import wandb

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()

    opt.layers_weights = [opt.vgg_w1, opt.vgg_w2, opt.vgg_w3, opt.vgg_w4, opt.vgg_w5]
    #opt.layers_weights = [1, 0.75, 0.2, 0.2, 0.2]
    # opt.layers_weights = [1, 1, 1, 1]
    opt.chosen_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    if torch.cuda.is_available() and not opt.is_cuda:
        print("WARNING: You have a CUDA device, consider removing --not_cuda")
    opt.is_cuda = opt.is_cuda and torch.cuda.is_available()
    opt.device = torch.device("cuda:0" if opt.is_cuda else "cpu")

    run_wandb = wandb.init(project='summer-project', config={})
    wandb.config.update(opt)

    # preprocess parameters
    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", opt.manual_seed)
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    opt.try_initial_guess = True if opt.try_initial_guess == 'true' else False
    opt.upsample_for_vgg = True if opt.upsample_for_vgg == 'true' else False
    if opt.upsample_for_vgg:
        print('Using upsampling for vgg')


    basename = os.path.basename(opt.image_path)
    basename = basename[:basename.rfind('.')]

    real_img = image_helpers.read_image(opt.image_path, opt.nc, opt.is_cuda)
    real_resized, scale_factor, total_scales = image_processing.preprocess_image(real_img, opt)

    opt.nzx = real_resized.shape[2]
    opt.nzy = real_resized.shape[0]
    out_dir = gen_unique_out_dir_path(opt.output_folder, basename, opt)


    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'params.txt'), 'w') as f:
        opt_dict = opt.__dict__
        opt_dict['device'] = opt_dict['device'].type
        json.dump(opt_dict, f)
    with open(os.path.join(opt.output_folder, basename, 'runs.txt'), 'a') as f:
        f.write(f'{os.path.basename(out_dir)} {opt}\n')

    try:
        Generators, z_opts, noise_amps, real_imgs = training.train(out_dir, real_resized, scale_factor, total_scales, opt)
        save_network(Generators, z_opts, noise_amps, real_imgs, out_dir)
        run_tests(Generators, z_opts, scale_factor, noise_amps, real_imgs, out_dir, opt)
        print('Done Training')
    except KeyboardInterrupt:
        print('done')

    run_wandb.finish()
