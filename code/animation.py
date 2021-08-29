from config import get_arguments
from training import *
import image_processing
import image_helpers
import imageio
import numpy as np
import output_handler
import random

def generate_gif(Generators,Zs,reals,NoiseAmp,opt, scale_factor, alpha=0.1,beta=0.9,start_scale=2, num_frames=100):
    """
    The funciton generate gif video from trained generators, and saves it.
    :param Generators: List of trained generators.
    :param Zs: Noise list of each scale, used for output image. Used for re-generating the image and random walk from there.
    :param reals: Real images list, scaled to the correct size.
    :param NoiseAmp: The noise amp in each new generation.
    :param opt: The config parameters of the script.
    :param alpha: The weight of the previous image in respect to new added differential
    :param beta: The weight of the difference between 2 following images in respect to random noise sample.
    :param start_scale: The scale from which the random walk start, instead of generating the image from the given noise, Zs.
    :param fps: The fps of the output gif. Along with :param num_frames:, sets the gif duration.
    :param num_frames: Number of frames in the output gif. Along with :param fps:, sets the gif duration.
    :return:
    """

    images_cur = []
    current_scale = 0
    #loop over all scales:
    for G,Z_opt,noise_amp,real in zip(Generators,Zs,NoiseAmp,reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        #pad_noise = 0
        #m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if current_scale == 0:
            z_rand = image_processing.generate_noise([1,nzx,nzy], device=opt.device)
            z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
            z_prev1 = 0.95*Z_opt +0.05*z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95*Z_opt +0.05*image_processing.generate_noise([opt.nc,nzx,nzy], device=opt.device)
            z_prev2 = Z_opt
        # Generate all the frames for the current scale:
        for i_frame in range(0,num_frames,1):
            # random walk:
            if current_scale == 0:
                z_rand = image_processing.generate_noise([1,nzx,nzy], device=opt.device)
                z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
            else:
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(image_processing.generate_noise([opt.nc,nzx,nzy], device=opt.device))

            z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            #add "history" so the animation will be smooth:
            if images_prev == []:
                I_prev = torch.full(Zs[0].shape, 0, device=opt.device)
            else:
                I_prev = images_prev[i_frame]
                I_prev = image_processing.resize(I_prev, 1 / scale_factor, opt.nc, opt.is_cuda)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                I_prev = m_image(I_prev)
            #generation of new images is only done for the relevant scales
            #TODO Seems a bit dumb to check *after* random walk calculation. maybe move to the top?
            if current_scale < start_scale:
                z_curr = Z_opt

            z_in = noise_amp*z_curr+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if (current_scale == len(Generators)-1):
                I_curr = image_helpers.denorm(I_curr).detach()
                I_curr = I_curr[0,:,:,:].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0)*255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        current_scale += 1

    return images_cur


if __name__ == '__main__':
    parser = get_arguments()
    #parser.add_argument('--animation_start_scale', type=int, help='generation start scale', default=2)
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--animation_initial_start_scale_sweep', help='Initial random scale for animation generation sweep.',type = int, default=0)
    parser.add_argument('--animation_final_start_scale_sweep', help='Final random scale for animation generation sweep.', type = int, default=3)
    parser.add_argument('--animation_initial_beta_sweep', help='Initial beta (weight of images diffrence vs new noise in random walk) for animation generation sweep.',type = float,default=0.8)
    parser.add_argument('--animation_final_beta_sweep', help='Final beta (weight of images diffrence vs new noise in random walk) for animation generation sweep.',type = float,default=0.95)
    parser.add_argument('--animation_alpha', help='Weight of of current image vs the diffrence vector.', type = float, default=0.1)
    parser.add_argument('--animation_fps', help='The fps of the output gif.',type = float, default=10)
    parser.add_argument('--animation_num_frames', help='The number of frames in the output gif.', type = int, default=100)
    #parser.add_argument('--alpha_animation', type=float, help='animation random walk first moment', default=0.1)
    #parser.add_argument('--beta_animation', type=float, help='animation random walk second moment', default=0.9)
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

    # dir2save = output_handler.gen_unique_out_dir_path(opt.output_folder, basename, opt)
    # opt.min_size = 20
    # opt.mode = 'animation_train'
    # real = functions.read_image(opt)
    # functions.adjust_scales2image(real, opt)
    # dir2trained_model = output_handler.gen_unique_out_dir_path(opt.output_folder, basename, opt)
    # if (os.path.exists(dir2trained_model)):
    #     Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    #     opt.mode = 'animation'
    # else:
    #     train(opt, Gs, Zs, reals, NoiseAmp)
    #     opt.mode = 'animation'
    Generators, z_opts, NoiseAmp, reals = output_handler.load_network(opt.trained_net_dir)
    for start_scale in range(opt.animation_initial_start_scale_sweep, opt.animation_final_start_scale_sweep, 1):
        for b in np.arange(opt.animation_initial_beta_sweep, opt.animation_final_beta_sweep, 0.05):
            frames = generate_gif(Generators, z_opts, reals, NoiseAmp, opt, scale_factor,
                                  alpha=opt.animation_alpha, beta=b, start_scale=start_scale,
                                  num_frames=opt.animation_num_frames)

            out_dir = os.path.join(opt.trained_net_dir, 'Animation', f"start_scale{start_scale}")
            os.makedirs(out_dir, exist_ok=True)
            gif_save_dir = os.path.join(out_dir, f"alpha={opt.animation_alpha}_beta=%.2f_fps={opt.animation_fps}.gif" % b)
            imageio.mimsave(gif_save_dir, frames, fps=opt.animation_fps)
