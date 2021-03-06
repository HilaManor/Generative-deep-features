from config import get_arguments
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tests
import os
import image_helpers
import output_handler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import image_processing
import plotting_helpers

from skimage import color



if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--patch_size', help='', type=int, default=15)
    parser.add_argument('--amount', type=int, default=50, help='the amount of images to '
                                                                'create and calculate the '
                                                                'standard variation on')
    opt = parser.parse_args()
    
    # Load the trained model parameters according to the params.txt file in the folder
    opt = output_handler.load_parameters(opt, opt.trained_net_dir)

    basename = os.path.basename(opt.image_path)
    basename = basename[:basename.rfind('.')]

    real_img = image_helpers.read_image(opt.image_path, opt.nc, opt.is_cuda)
    real_resized, scale_factor, total_scales = image_processing.preprocess_image(real_img, opt)
    opt.nzx = real_resized.shape[2]
    opt.nzy = real_resized.shape[0]

    Generators, z_opts, NoiseAmp, reals = output_handler.load_network(opt.trained_net_dir)
    n=2
    real_resized = image_processing.resize(real_resized, pow(scale_factor, (N - n + 1)), opt.nc, opt.is_cuda)
    
    # Fit KNN model
    nn = NearestNeighbors(n_neighbors=1)
    real_patches = []
    im = color.rgb2gray(plotting_helpers.convert_im(real_resized))
    for y in range(im.shape[0] - opt.patch_size):
        for x in range(im.shape[1] - opt.patch_size):
            patch = im[y:y + opt.patch_size, x:x + opt.patch_size]
            real_patches.append(patch.ravel())
            #print(f'Patch {y*(im.shape[0] - opt.patch_size)+ x}/{(im.shape[0] - opt.patch_size)*(im.shape[1] - opt.patch_size)}', end='\r')
    nn.fit(np.stack(real_patches, axis=0))
    # Generate images
    ims = []
    out_dir = os.path.join(opt.trained_net_dir, 'Edges_experiments')
    if os.path.isdir(out_dir):
        for f in os.listdir(out_dir):
            filepath = os.path.join(out_dir, f)
            ims.append(plt.imread(filepath))
    else:
        os.makedirs(out_dir, exist_ok=True)
        for i in range(opt.amount):
            out = tests.generate_random_sample(Generators, z_opts, scale_factor, NoiseAmp,
                                           reals, opt=opt)
            plotting_helpers.save_im(out[-1], out_dir, f"seed_{opt.manual_seed}_im_{i}",
                                     convert=True)
            #ims.append((256*color.rgb2gray(plotting_helpers.convert_im(out[-1]))).astype('uint8'))
            #ims.append(color.rgb2gray(plotting_helpers.convert_im(out[-1])))
            
            #color.rgb2gray(plotting_helpers.convert_im(out[-1]))
            
            im = color.rgb2gray(plotting_helpers.convert_im(out[-1]))
            dist_map = np.zeros(im.shape)
            for y in range(im.shape[0] - opt.patch_size):
                print(f'image: {i}/{opt.amount}\t\t'
                      f'Patch {y*(im.shape[0] - opt.patch_size)}/{(im.shape[0] - opt.patch_size)*(im.shape[1] - opt.patch_size)}', 
                      end='\r')
                for x in range(im.shape[1] - opt.patch_size):
                    patch = im[y:y + opt.patch_size, x:x + opt.patch_size]
                    dist, ind = nn.kneighbors(np.expand_dims(patch.ravel(), axis=0), 1, return_distance=True)
                    dist_map[np.unravel_index(ind, dist_map.shape)] = dist
            ims.append(dist_map)
            
    ims = np.stack(ims, axis=-1)
    ims_avg = np.mean(ims, axis=2)
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(ims_var/real_var, cmap='jet')#, vmin=0, vmax=1.5)
    #plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)# ticks=[0,0.5,1,1.5], cax=cax)
    plt.savefig(os.path.join(opt.trained_net_dir, f'edges_exper_avg_n{opt.amount}_p{opt.patch_size}.png'))
    