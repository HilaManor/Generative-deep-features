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
    
    
    # Fit KNN model
    real_patches = []
    im = color.rgb2gray(plotting_helpers.convert_im(real_resized))
    for y in range(im.shape[0] - opt.patch_size + 1):
        for x in range(im.shape[1] - opt.patch_size + 1):
            patch = im[y:y + opt.patch_size, x:x + opt.patch_size]
            real_patches.append(patch.ravel())
            #print(f'Patch {y*(im.shape[0] - opt.patch_size)+ x}/{(im.shape[0] - opt.patch_size)*(im.shape[1] - opt.patch_size)}', end='\r')
    real_patches_mat = torch.from_numpy(np.stack(real_patches, axis=0)).to(device=opt.device)
    
    # Generate images
    ims_mse = []
    ims_edge = []
    out_dir = os.path.join(opt.trained_net_dir, 'Edges_experiments_pytorch')
    #if os.path.isdir(out_dir):
    #    for f in os.listdir(out_dir):
    #        filepath = os.path.join(out_dir, f)
    #        #ims.append(plt.imread(filepath))
    #else:
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
        dist_map_mse = np.zeros((im.shape[0] - opt.patch_size + 1, im.shape[1] - opt.patch_size + 1))
        dist_map_edge = np.zeros((im.shape[0] - opt.patch_size + 1, im.shape[1] - opt.patch_size + 1 ))
        
        for y in range(im.shape[0] - opt.patch_size + 1):
            print(f'image: {i}/{opt.amount}\t\t'
                  f'Patch {y*(im.shape[0] - opt.patch_size + 1)}/{(im.shape[0] - opt.patch_size + 1)*(im.shape[1] - opt.patch_size + 1)}', 
                  end='\r')
            for x in range(im.shape[1] - opt.patch_size + 1):
                patch = im[y:y + opt.patch_size, x:x + opt.patch_size]
                mse = torch.mean((torch.from_numpy(patch.ravel()).to(device=opt.device) - real_patches_mat)**2, axis=1)
                mse_ind = torch.argmin(mse)
                mse_2d_ind = np.unravel_index(mse_ind.cpu().numpy(), dist_map_mse.shape) 
                dist_to_edge = min([mse_2d_ind[0], 
                                    np.abs(mse_2d_ind[0] - (im.shape[0] - opt.patch_size)), 
                                    mse_2d_ind[1],
                                    np.abs(mse_2d_ind[1] - (im.shape[1] - opt.patch_size))])
                                    
                dist_map_mse[mse_2d_ind] = mse[mse_ind].cpu().numpy()
                dist_map_edge[mse_2d_ind] = dist_to_edge
        ims_mse.append(dist_map_mse)
        ims_edge.append(dist_map_edge)
            
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
    plt.savefig(os.path.join(opt.trained_net_dir, f'edges_exper_avgmse_n{opt.amount}_p{opt.patch_size}.png'))
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(ims_edge_avg)
    #im = ax.imshow(ims_var/real_var, cmap='jet')#, vmin=0, vmax=1.5)
    #plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)# ticks=[0,0.5,1,1.5], cax=cax)
    plt.savefig(os.path.join(opt.trained_net_dir, f'edges_exper_avgedge_n{opt.amount}_p{opt.patch_size}.png'))
    
