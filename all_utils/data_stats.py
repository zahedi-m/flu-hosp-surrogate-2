from pathlib import Path
import torch

from all_utils.data_utils import FeatureDataset, collate_fn

import numpy as np

def get_z_score_stat(dataPath, x_dim, xt_dim, y_dim, seq_len, NUM_COMP, populations, category="train", batch_size= 64):
    
    dataPath= Path(dataPath) if not isinstance(dataPath, Path) else dataPath

    dataset= FeatureDataset(dataPath, seq_len, category="train", populations=populations)
#
    loader= torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle=False, collate_fn =collate_fn)
    yd= y_dim*NUM_COMP
    xsum= torch.zeros(x_dim)
    xsum2= torch.zeros(x_dim)
    #
    ysum= torch.zeros(yd)
    ysum2= torch.zeros(yd)
    #
    x_counts=0
    xt_counts=0
    y_counts=0

    # x:[B, #nodes, x_dim]
    # y:[B, L, ydim]
    # y0:[B, y_dim]

    for batch_idx, (x, xt, y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev, y0) in enumerate(loader):
        y_hosp_inc= torch.cat([torch.zeros(*y0.size()).unsqueeze(1), y_hosp_inc], dim=1)
        y_hosp_prev= torch.cat([torch.zeros(*y0.size()).unsqueeze(1), y_hosp_prev], dim=1)
        y_latent_inc= torch.cat([torch.zeros(*y0.size()).unsqueeze(1), y_latent_inc], dim=1)
        y_latent_prev= torch.cat([y0.unsqueeze(1), y_latent_prev], dim=1)
        y= torch.concat([y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev], dim=-1)
        xsum[xt_dim:] += x.sum(dim=(0, 1))
        xsum2[xt_dim:] += (x**2.0).sum(dim=(0, 1))
        # temporal xt
        xsum[:xt_dim] += xt.sum(dim=(0, 1, 2))
        xsum2[:xt_dim] += (xt**2.0).sum(dim=(0, 1, 2))
        ysum += y.sum(dim=(0, 1))
        ysum2 += (y**2.0).sum(dim=(0, 1))
        #
        x_counts+= np.prod(x.shape[:-1])
        xt_counts+= np.prod(xt.shape[:-1]) 
        y_counts+=np.prod(y.shape[:-1])
    # 
    x_mean= np.zeros_like(xsum)
    x_std= np.zeros_like(xsum)

    x_mean[xt_dim:]= xsum[xt_dim:].numpy()/x_counts
    x_mean[:xt_dim]= xsum[:xt_dim].numpy()/xt_counts

    x_std[xt_dim:]= np.sqrt(xsum2[xt_dim:].numpy()/x_counts-x_mean[xt_dim:]**2.0)
    x_std[:xt_dim]= np.sqrt(xsum2[:xt_dim].numpy()/xt_counts-x_mean[:xt_dim]**2.0)
    
    x_std[x_std==0.0]=1e-8
    #
    y_mean= ysum.numpy()/y_counts
    y_std= np.sqrt(ysum2.numpy()/y_counts- y_mean**2.0)
    y_std[y_std==0.0]=1e-8
    #
    return x_mean, x_std, y_mean, y_std

