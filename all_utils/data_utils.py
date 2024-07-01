import torch
import numpy as np

from typing import Callable, Literal

from functools import partial
from pathlib import Path

import pandas as pd
import pickle
from all_utils.input_constructors import construct_x, construct_initial_conditions, construct_temporal_features
from all_utils.save_utils import make_x_filename, make_xt_filename, make_y_filename, make_y0_filename

def load_graph_data(dataPath):

    if not isinstance(dataPath, Path):
        dataPath= Path(dataPath)

    df= pd.read_csv(dataPath.joinpath("weighted_edge_list.csv"), sep=",", header=None, names=["source", "target", "weights"])

    edge_index= df[["source", "target"]].values.T
    edge_weight= df["weights"].values
    return edge_index, edge_weight


def get_filenames(dataPath, category:Literal["train", "val", "test"]):  
    """
        category: "train", "val", "test"
    """
    src= Path(dataPath) if not isinstance(dataPath, Path) else dataPath

    x_filenames= sorted(list( src.joinpath("_".join(["x", category])).glob("*.npy") ))
    xt_filenames= sorted(list( src.joinpath("_".join(["xt", category])).glob("*.npy") ))

    y_inc_filenames= sorted(list( src.joinpath("_".join(["y", "inc", category])).glob("*.npy") ))
    y_prev_filenames= sorted(list( src.joinpath("_".join(["y", "prev", category])).glob("*.npy") ))

    y0_filenames=sorted(list( src.joinpath("_".join(["y0", category])).glob("*.npy") ))

    return x_filenames, xt_filenames, y_inc_filenames, y_prev_filenames, y0_filenames

def get_datasets(metaPath, dataPath, srcPath, x_col_names, frac_pops_names, initial_col_names, seq_len, num_nodes, population_csv_path, POPULATION_SCALER):

    dataPath= Path(dataPath) if not isinstance(dataPath, Path) else dataPath
    metaPath= Path(metaPath) if not isinstance(metaPath, Path) else metaPath
    srcPath= Path(srcPath) if not isinstance(srcPath, Path) else srcPath
   
    # to account for initial values
    populations= load_population_data(population_csv_path)
    train_dataset= FeatureDataset(dataPath, seq_len, "train", populations)

    val_dataset= FeatureDataset(dataPath, seq_len, "val", populations)

    pool_dataset= PoolDataset(metaPath.joinpath("x_df.csv"), x_col_names, frac_pops_names, initial_col_names, seq_len, num_nodes, populations, POPULATION_SCALER)

    return train_dataset, val_dataset, pool_dataset

class PoolDataset(torch.utils.data.Dataset):

    def __init__(self, meta_filename, x_col_names, frac_pops_names, initial_col_names, seq_len, num_nodes, populations, POPULATION_SCALER, x_transform:Callable=None, y_transform:Callable=None, ):
        
        self.x_df= read_meta_df(meta_filename)

        self.x_col_names= x_col_names
        self.frac_pops_names= frac_pops_names
        self.initial_col_names= initial_col_names

        self.seq_len= seq_len
        self.max_seq_len= seq_len+1 # account for initial condition
        self.num_nodes= num_nodes
        
        self.pop_data= populations
        self.POPULATION_SCALER =POPULATION_SCALER

        self.x_transform= x_transform
        self.y_transform= y_transform

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, i):
        """
            @return :
            x:[B, #nodes, x_dim]
            xt:[B, #nodes, L, xt_dim]
        """
        df= self.x_df.iloc[i]

        x= df.loc[self.x_col_names].values

        x= construct_x(df, self.x_col_names, self.frac_pops_names, self.pop_data, self.POPULATION_SCALER)[0]

        xt= construct_temporal_features(df, self.seq_len, self.num_nodes)[:, :self.max_seq_len, ...]

        # add batch axisS
        if np.ndim(x)<3:
            x= x[np.newaxis, ...]
        
        if np.ndim(xt)< 4:
            xt= xt[np.newaxis, ...]

        df_y0= df.loc[self.initial_col_names]

        y0= construct_initial_conditions(df_y0, self.pop_data, self.POPULATION_SCALER)[0]

         # add batch dimension
        if np.ndim(y0)<2:
            y0= y0[np.newaxis, ...]

        if self.x_transform:
            x= self.x_transform(x)
        
        if self.y_transform :
            y0= self.y_transform(y0)
    
        return x.astype(np.float32), xt.astype(np.float32), y0.astype(np.float32), i
    
    def update_transforms(self, x_transform, y_transform):
        self.x_transform= x_transform 
        self.y_transform= y_transform

class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataPath,  seq_len, category:Literal["train", "val", "test"], populations, x_transform=None, y_hosp_inc_transform=None, y_hosp_prev_transform=None, y_latent_inc_transform=None, y_latent_prev_transform=None):
        
        dataPath= Path(dataPath) if not isinstance(dataPath, Path) else dataPath
        self.max_seq_len= seq_len+1 # initial condition
        
        self.x_path= dataPath.joinpath("x_"+category)
        #temporal features
        self.xt_path=dataPath.joinpath("xt_"+category)

        self.y_inc_path= dataPath.joinpath("y_inc_"+category)
        self.y_prev_path= dataPath.joinpath("y_prev_"+category)
        self.y0_path= dataPath.joinpath("y0_"+category)

        self.x_filenames= sorted(list(self.x_path.glob("*.npy")))
        self.xt_filenames= sorted(list(self.xt_path.glob("*.npy")))

        self.y_inc_filenames= sorted(list(self.y_inc_path.glob("*.npy")))
        self.y_prev_filenames= sorted(list(self.y_prev_path.glob("*.npy")))
        self.y0_filenames= sorted(list(self.y0_path.glob("*.npy")))

        self.pop_data= populations

        self.x_transform= x_transform
        self.y_hosp_inc_transform= y_hosp_inc_transform
        self.y_hosp_prev_transform= y_hosp_prev_transform
        self.y_latent_inc_transform= y_latent_inc_transform
        self.y_latent_prev_transform= y_latent_prev_transform
       
    def __len__(self):
        return len(self.x_filenames)
    
    def __getitem__(self, i:int):
        """
            @return:
            x:[num_sims, L, #nodes, xdim],  
            xt:[num_sims, L, #nodes, xt_dim]
            y:[num_sims, L, y_dim], 
            y0:[num_sims, y_dim], index
        """
        x= np.load(self.x_filenames[i]).astype(np.float64)
        xt= np.load(self.xt_filenames[i]).astype(np.float64)[:, :self.max_seq_len, ...]
        y0= np.load(self.y0_filenames[i]).astype(np.float64)
        y_inc= np.load(self.y_inc_filenames[i]).astype(np.float64)[:, :self.max_seq_len,...]
        y_hosp_inc, y_latent_inc= np.split(y_inc, 2, axis=-1)
        y_prev= np.load(self.y_prev_filenames[i]).astype(np.float64)[:, :self.max_seq_len,...]
        y_hosp_prev, y_latent_prev= np.split(y_prev, 2, axis=-1)
        if self.x_transform:
            x= self.x_transform(x)
        if self.y_latent_prev_transform:
            y_latent_prev= self.y_latent_prev_transform.transform(y_latent_prev)
            y0= self.y_latent_prev_transform.transform(y0)
        if self.y_latent_inc_transform:
            y_latent_inc= self.y_latent_inc_transform.transform(y_latent_inc)
        if self.y_hosp_inc_transform:
            y_hosp_inc= self.y_hosp_inc_transform.transform(y_hosp_inc)
        if self.y_hosp_prev_transform:
            y_hosp_prev= self.y_hosp_prev_transform.transform(y_hosp_prev)
           
        return (x.astype(np.float32), xt.astype(np.float32), y_hosp_inc[:, 1:, ...].astype(np.float32), 
                y_hosp_prev[:, 1:, ...].astype(np.float32), y_latent_inc[:, 1:, ...].astype(np.float32), 
                y_latent_prev[:, 1:, ...].astype(np.float32), y0.astype(np.float32), i)
     
    def update_transforms(self, x_transform, y_hosp_inc_transform, y_hosp_prev_transform, y_latent_inc_transform, y_latent_prev_transform):
        self.x_trandform = x_transform 
        self.y_hosp_inc_transform= y_hosp_inc_transform
        self.y_hosp_prev_transform= y_hosp_prev_transform
        self.y_latent_inc_transform= y_latent_inc_transform
        self.y_latent_prev_transform= y_latent_prev_transform

    def update_dataset(self):
        self.x_filenames= sorted(list(self.x_path.glob("*.npy")))
        self.xt_filenames= sorted(list(self.xt_path.glob("*.npy")))
        self.y_inc_filenames= sorted(list(self.y_inc_path.glob("*.npy")))
        self.y_prev_filenames= sorted(list(self.y_prev_path.glob("*.npy")))
        self.y0_filenames= sorted(list(self.y0_path.glob("*.npy")))
#
def collate_fn(batch, device=None):
    x, xt, y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev, y0, _= zip(*batch)
   
    x= torch.cat([torch.FloatTensor(xx) for xx in x], dim=0)
    xt= torch.cat([torch.FloatTensor(xx) for xx in xt], dim=0)
    y_hosp_inc=torch.cat([torch.FloatTensor(yy) for yy in y_hosp_inc], dim=0)
    y_hosp_prev=torch.cat([torch.FloatTensor(yy) for yy in y_hosp_prev], dim=0)
    y_latent_inc=torch.cat([torch.FloatTensor(yy) for yy in y_latent_inc], dim=0)
    y_latent_prev=torch.cat([torch.FloatTensor(yy) for yy in y_latent_prev], dim=0)
    y0= torch.cat([torch.FloatTensor(yy) for yy in y0], dim=0)
    perm_indx= np.random.permutation(x.shape[0])
    if device is not None:
        x, xt, y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev, y0= x.to(device), xt.to(device), y_hosp_inc.to(device), y_hosp_prev.to(device), y_latent_inc.to(device), y_latent_prev.to(device), y0.to(device)

    return x[perm_indx, ...], xt[perm_indx, ...], y_hosp_inc[perm_indx, ...], y_hosp_prev[perm_indx, ...], y_latent_inc[perm_indx, ...], y_latent_prev[perm_indx, ...], y0[perm_indx, ...]
#
def pool_collate_fn(batch, device=None):

    x, xt, y0, pool_indices= zip(*batch)
    
    x= torch.cat([torch.FloatTensor(xx) for xx in x], dim=0)
    xt= torch.cat([torch.FloatTensor(xx) for xx in xt], dim=0)
    
    y0= torch.cat([torch.FloatTensor(yy) for yy in y0], dim=0)
    
    pool_indices= np.array(pool_indices , dtype= np.int64)

    if device is not None:
        x, xt, y0, pool_indices= x.to(device), xt.to(device), y0.to(device), pool_indices

    return x, xt, y0, pool_indices
#

def get_z_score_transform(x_mean, x_std, y_mean, y_std):
    x_transform = partial(standardize, mean= x_mean, std= x_std)
    y_transform = partial(standardize, mean= y_mean, std= y_std)
    return x_transform, y_transform

def standardize(data:np.ndarray, mean, std):
    std[std==0.0]=1e-8
    #TODO use assert
    #assert np.all()
    return (data-mean)/std

#
def read_meta_df(meta_filename, use_cols=None):
    """ 
        use_cols=None : read all columns
    """
    if use_cols is not None:
        df= pd.read_csv(meta_filename, usecols=use_cols)
    else:
        df= pd.read_csv(meta_filename, parse_dates=["starting_date"])

    if use_cols is None or "run_ids" in use_cols:   
        df["run_ids"]= df["run_ids"].apply(eval)

    return df
#

def add_pop_data_to_x(x, pop_data):
    """ 
    add population data to input x 
        @input:
            x:[B, #nodes, x_dim]
            x:(R0, days, pops_frac) 
            pop_data: popluation data
        @return:
        [B, 51, x_dim+#frac_prev_pop]
    """
    x_1= x[..., :2]
    x_frac_pops= x[..., 2:]

    #[B, 51, #comp]
    x_pops= np.einsum("ikj,k->ikj", x_frac_pops.astype(np.float64), pop_data)

    return np.concatenate([x_1, x_pops], axis=-1)
#

def remove_extra_train_files(path, last_iter_train_file_id):
    
    x_filenames, xt_filenames, y_inc_filenames, y_prev_filenames, y0_filenames= get_filenames(path, category="train")
    
    x_parent, xt_parent, y_inc_parent, y_prev_parent, y0_parent= x_filenames[0].parent, xt_filenames[0].parent, y_inc_filenames[0].parent, y_prev_filenames[0].parent, y0_filenames[0].parent

    #just keep name and remove full path
    x_filenames= [f.name for f in x_filenames]
    xt_filenames= [f.name for f in xt_filenames]

    y_inc_filenames= [f.name for f in y_inc_filenames]
    y_prev_filenames= [f.name for f in y_prev_filenames]
    y0_filenames= [f.name for f in y0_filenames]
    

    last_iter_x_filenames=[make_x_filename(idx) for idx in last_iter_train_file_id]
    last_iter_xt_filenames=[make_xt_filename(idx) for idx in last_iter_train_file_id]

    last_iter_y_inc_filenames= [make_y_filename(idx, "y_inc") for idx in last_iter_train_file_id]
    last_iter_y_prev_filenames= [make_y_filename(idx, "y_prev") for idx in last_iter_train_file_id]
    
    last_iter_y0_filenames= [make_y0_filename(idx) for idx in last_iter_train_file_id]
    
    # extra_x_filenames= set(x_filenames).difference(set(last_iter_x_filenames))
    # extra_xt_filenames= set(xt_filenames).difference(set(last_iter_xt_filenames))
    

    # extra_y_inc_filenames= set(y_inc_filenames).difference(set(last_iter_y_inc_filenames))
    # extra_y_prev_filenames= set(y_prev_filenames).difference(set(last_iter_y_prev_filenames))
    # extra_y0_filenames= set(y0_filenames).difference(set(last_iter_y0_filenames))
    
    #deleting files
    [Path(x_parent).joinpath(f).unlink() for f in last_iter_x_filenames]
    [Path(xt_parent).joinpath(f).unlink() for f in last_iter_xt_filenames]
    
    [Path(y_inc_parent).joinpath(f).unlink() for f in last_iter_y_inc_filenames]
    [Path(y_prev_parent).joinpath(f).unlink() for f in last_iter_y_prev_filenames]
    [Path(y0_parent).joinpath(f).unlink() for f in last_iter_y0_filenames]


def load_population_data(population_csv_path):
    return pd.read_csv(population_csv_path).sort_values("country_id")["population"].values

class ZNormalize:
    def __init__(self, mean, std):
        self.eps= 1e-8
        if isinstance(mean, torch.Tensor):
            self.mean= mean.detach().numpy().astype(np.float64)
        elif isinstance(mean, np.ndarray) or None:
            self.mean= mean.astype(np.float64)
        else:
            raise TypeError("input must be numpy array or torch Tensor")
        
        if isinstance(std, torch.Tensor):
            self.std= std.detach.numpy().astype(np.float64)
        elif isinstance(std, np.ndarray) or None:
            self.std= std.astype(np.float64)
        else:
            raise TypeError("input must be numpy array or torch Tensor")
        
    def transform(self, y):
        if isinstance(y, torch.Tensor):
            torch_mean= torch.from_numpy(self.mean).to(y.device).to(y.dtype)
            torch_std= torch.from_numpy(self.std).to(y.device).to(y.dtype)
            return (y-torch_mean)/(torch_std+self.eps)
        elif isinstance(y, np.ndarray):
            return (y-self.mean)/(self.std+self.eps)
        else:
            raise TypeError("input must be numpy array or torch Tensor")
    def upddate_stats(self, mean, std):
        self.mean= mean
        self.std= std
    def get_params(self):
        return self.mean, self.std
    
    def __call__(self, y):
        self.transform(y)
