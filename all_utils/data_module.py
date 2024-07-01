from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

import torch

import numpy as np

import os, glob
from functools import partial

from typing import Optional, Callable

from torch.utils.data.sampler import SubsetRandomSampler


class FluDataModule(LightningDataModule):

    def __init__(self, dataPath, train_batch_size:int, val_batch_size:int):
        super().__init__()

        self.dataPath= dataPath
 
        self.train_batch_size=train_batch_size
        self.val_batch_size= val_batch_size

        # train_dataset
        self.x_train_filenames=sorted(glob.glob(os.path.join(dataPath,  "x_train", "*.npy")))
        self.y_train_filenames= sorted(glob.glob(os.path.join(dataPath,  "y_train", "*.npy")))

        #indices= np.random.choice(len(self.x_train_filenames), size=2000)

        #self.x_train_filenames= [self.x_train_filenames[idx] for idx in indices]
        #self.y_train_filenames= [self.y_train_filenames[idx] for idx in indices]

        # val dataset
        self.x_val_filenames= sorted(glob.glob(os.path.join(dataPath, "x_val", "*.npy")))
        self.y_val_filenames= sorted(glob.glob(os.path.join(dataPath, "y_val", "*.npy")))


       
        #indices= np.random.choice(len(self.x_val_filenames), size=20)
    
        #self.x_val_filenames= [self.x_val_filenames[idx] for idx in indices]

        #self.y_val_filenames= [self.y_val_filenames[idx] for idx in indices]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
                        
        stats= np.load(os.path.join(self.dataPath, "stats.npz"))
        self.x_transform = partial(standardize, mean= stats["x_mean"], std= stats["x_std"])
        self.y_transform = partial(standardize, mean= stats["y_mean"], std= stats["y_std"])

    def train_dataloader(self):
        train_dataset= FeatureDataset(self.x_train_filenames, self.y_train_filenames, x_transform= self.x_transform, y_transform= self.y_transform)

        train_dataloader= DataLoader(train_dataset, collate_fn=self.collate_fn,
                            batch_size=self.train_batch_size, num_workers=4, drop_last=True, shuffle=True)        
        
        return train_dataloader

    def val_dataloader(self):
        val_dataset=FeatureDataset(self.x_val_filenames, self.y_val_filenames, x_transform=self.x_transform, y_transform= self.y_transform)

        feature_dataloader= DataLoader(val_dataset, batch_size=self.val_batch_size, num_workers=1, collate_fn=self.collate_fn, drop_last=True)        
    
        return feature_dataloader
    
    
    @staticmethod
    def collate_fn(batch):
        x, y, y0= zip(*batch)
        x= torch.cat([torch.FloatTensor(xx) for xx in x], dim=0)
        y=torch.cat([torch.FloatTensor(yy) for yy in y], dim=0)
        y0= torch.cat([torch.FloatTensor(yy) for yy in y0], dim=0)

        #indices= np.random.permutation(x.shape[0])

        return x, y, y0

#
def scale_transform(x, scale_factor):
        x= np.where(x<0, 0, x)
        x/=scale_factor
        return x

def log_transform (x):
    x= np.where(x<0, 0.0, x)
    return np.log(x+1.0)
#
def standardize(data:np.ndarray, mean, std):
    return (data-mean)/(std+ np.finfo(data.dtype).eps)
     
class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(self, x_filenames:list, y_filenames:list, x_transform:Callable=None, y_transform:Callable=None):
        
        self.x_filenames= x_filenames
        self.y_filenames= y_filenames

        self.x_transform = x_transform
        self.y_transform= y_transform 
       
    def __len__(self):
        return len(self.x_filenames)
    
    def __getitem__(self, i:int):
        """
            @return:
            # x:[num_sims, L, #nodes, xdim],  y:[num_sims, L, y_dim], y0:[num_sims, y_dim]
        """

        x= np.load(self.x_filenames[i])
        y=np.load(self.y_filenames[i])

        if self.y_transform:
            y= self.y_transform(y)
        if self.x_transform:
            x= self.x_transform(x)
         
        return x.astype(np.float32), y[:,1:, ...].astype(np.float32) , y[:, 0,...].astype(np.float32)

#
class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, edgelist, nodelist):

        self.edgelist= edgelist
        self.nodelist= nodelist

        self.preprocess()

    def preprocess(self):
        node2zipcode=dict(zip(range(len(self.nodelist)), np.squeeze(self.nodelist.values)))
        
        zipcode2node=dict(zip(node2zipcode.values(), node2zipcode.keys()))

        self.edgelist.replace({"source_county_fips_code":zipcode2node, "target_county_fips_code":zipcode2node}, inplace=True)

        
        nodes=np.unique(self.edgelist[["source_county_fips_code", "target_county_fips_code"]].values)

        self.num_nodes=nodes.shape[0]

        self.edge_index= self.edgelist.values[:, :2].T
        self.edge_attr= self.edgelist.values[:, -1]

    def __len__(self):
        return 28

    def __getitem__(self, i):
        return self.edge_index.astype(np.int64), self.edge_attr.astype(np.float32)

