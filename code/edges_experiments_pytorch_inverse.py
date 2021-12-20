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
import torch


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
    
    real_im = color.rgb2gray(plotting_helpers.convert_im(real_resized))
    
    
    # Generate images
    ims_mse = []
    ims_edge = []
    out_dir = os.path.join(opt.trained_net_dir, 'Edges_experiments_pytorch_real')
    #if os.path.isdir(out_dir):
    #    for f in os.listdir(out_dir):
    #        filepath = os.path.join(out_dir, f)
    #        #ims.append(plt.imread(filepath))
    #else:
    os.makedirs(out_dir, exist_ok=True)
    out_dir_mse = os.path.join(opt.trained_net_dir, 'Edges_experiments_pytorch_maps_mse_real')
    out_dir_edge = os.path.join(opt.trained_net_dir, 'Edges_experiments_pytorch_maps_edge_real')
    os.makedirs(out_dir_mse, exist_ok=True)
    os.makedirs(out_dir_edge, exist_ok=True)
    for i in range(opt.amount):
        out = tests.generate_random_sample(Generators, z_opts, scale_factor, NoiseAmp,
                                       reals, opt=opt)
        plotting_helpers.save_im(out[-1], out_dir, f"seed_{opt.manual_seed}_im_{i}",
                                 convert=True)
        #ims.append((256*color.rgb2gray(plotting_helpers.convert_im(out[-1]))).astype('uint8'))
        #ims.append(color.rgb2gray(plotting_helpers.convert_im(out[-1])))
        
        #color.rgb2gray(plotting_helpers.convert_im(out[-1]))
        
        im = color.rgb2gray(plotting_helpers.convert_im(out[-1]))
        
        # create patches matrix for KNN
        gen_patches = []
        for y in range(im.shape[0] - opt.patch_size + 1):
            for x in range(im.shape[1] - opt.patch_size + 1):
                patch = im[y:y + opt.patch_size, x:x + opt.patch_size]
                gen_patches.append(patch.ravel())
                #print(f'patch {y*(im.shape[0] - opt.patch_size)+ x}/{(im.shape[0] - opt.patch_size)*(im.shape[1] - opt.patch_size)}', end='\r')
        gen_patches_mat = torch.from_numpy(np.stack(gen_patches, axis=0)).to(device=opt.device)
        
        
        dist_map_mse = np.zeros((real_im.shape[0] - opt.patch_size + 1, real_im.shape[1] - opt.patch_size + 1))
        dist_map_edge = np.zeros((real_im.shape[0] - opt.patch_size + 1, real_im.shape[1] - opt.patch_size + 1 ))
        for y in range(real_im.shape[0] - opt.patch_size + 1):
            print(f'image: {i}/{opt.amount}\t\t'
                  f'Patch {y*(real_im.shape[0] - opt.patch_size + 1)}/{(real_im.shape[0] - opt.patch_size + 1)*(real_im.shape[1] - opt.patch_size + 1)}', 
                  end='\r')
            for x in range(real_im.shape[1] - opt.patch_size + 1):
                patch = real_im[y:y + opt.patch_size, x:x + opt.patch_size]
                mse = torch.mean((torch.from_numpy(patch.ravel()).to(device=opt.device) - gen_patches_mat)**2, axis=1)
                mse_ind = torch.argmin(mse)
                mse_2d_ind = np.unravel_index(mse_ind.cpu().numpy(), dist_map_mse.shape) 
                dist_to_edge = min([mse_2d_ind[0], 
                                    np.abs(mse_2d_ind[0] - (real_im.shape[0] - opt.patch_size)), 
                                    mse_2d_ind[1],
                                    np.abs(mse_2d_ind[1] - (real_im.shape[1] - opt.patch_size))])
                                    
                dist_map_mse[mse_2d_ind] = mse[mse_ind].cpu().numpy()
                dist_map_edge[mse_2d_ind] = dist_to_edge
        
        ims_mse.append(dist_map_mse)
        ims_edge.append(dist_map_edge)
        plotting_helpers.save_im(dist_map_mse, out_dir_mse, f"seed_{opt.manual_seed}_im_{i}", convert=False)
        plotting_helpers.save_im(dist_map_edge, out_dir_edge, f"seed_{opt.manual_seed}_im_{i}", convert=False)
            
    ims_mse = np.stack(ims_mse, axis=-1)
    ims_edge = np.stack(ims_edge, axis=-1)
    ims_mse_avg = np.mean(ims_mse, axis=2)
    ims_edge_avg = np.mean(ims_edge, axis=2)
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(ims_mse_avg)
    #im = ax.imshow(ims_var/real_var, cmap='jet')#, vmin=0, vmax=1.5)
    #plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)# ticks=[0,0.5,1,1.5], cax=cax)
    plt.savefig(os.path.join(opt.trained_net_dir, f'real_edges_exper_avgmse_n{opt.amount}_p{opt.patch_size}.png'))
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(ims_edge_avg)
    #im = ax.imshow(ims_var/real_var, cmap='jet')#, vmin=0, vmax=1.5)
    #plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)# ticks=[0,0.5,1,1.5], cax=cax)
    plt.savefig(os.path.join(opt.trained_net_dir, f'real_edges_exper_avgedge_n{opt.amount}_p{opt.patch_size}.png'))
    
