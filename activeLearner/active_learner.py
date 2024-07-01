import torch

from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.profilers import PyTorchProfiler

import numpy as np
from pathlib import Path

import yaml
import pandas as pd
from itertools import chain
import pytorch_lightning as pl
from .active_dataset import ActiveLearningData

import torch.nn as nn

from typing import Type

from all_utils.data_utils import collate_fn, get_z_score_transform, read_meta_df, remove_extra_train_files

from all_utils.input_constructors import  construct_initial_conditions, construct_x, construct_temporal_features

from all_utils.save_utils import save_x_data, save_y0_data, save_xt_data

from all_utils.data_stats import get_z_score_stat

from all_utils.torch_utils import load_model, load_active_data
from all_utils.bucket_retriever import map_runid_to_fname
from all_utils.data_utils import load_population_data, ZNormalize
import cProfile, pstats

class ActiveLearner:
    model: nn.Module
    train_dataset: Dataset
    val_dataset: Dataset
    pool_dataset: Dataset
    aquisition:Type
    dataRetriever:Type
    params: dict
    def __init__(self, model, 
                x_dim,
                xt_dim,
                y_dim,
                seq_len,
                num_nodes, 
                train_dataset, 
                val_dataset, 
                pool_dataset, 
                acquisition, 
                dataRetriever,  
                max_epochs,
                train_batch_size,
                train_num_workers,
                initial_train_size,
                val_batch_size,
                val_num_workers,
                lr,
                dataPath,
                x_col_names, 
                frac_pops_names,
                device, 
                populaiton_csv_path, 
                POPULATION_SCALER, 
                batch_size_stat_compute,
                NUM_COMP,
                **kwargs):
        
        self.x_dim= x_dim
        self.xt_dim= xt_dim
        self.y_dim=y_dim
        self.seq_len=seq_len
        self.num_nodes= num_nodes
        self.model= model
        self.train_dataset= train_dataset
        self.pool_dataset= pool_dataset
        self.val_dataset= val_dataset
        self.acquisition= acquisition
        self.dataRetriever= dataRetriever
       
        self.max_epochs= max_epochs
        self.train_batch_size= train_batch_size
        self.train_num_workers= train_num_workers
        self.initial_train_size= initial_train_size
        self.val_batch_size= val_batch_size
        self.val_num_workers= val_num_workers
        self.lr= lr
        self.dataPath= dataPath
        self.x_col_names= x_col_names
        self.frac_pops_names= frac_pops_names
        self.device= device
        self.POPULATION_SCALER= POPULATION_SCALER
        self.batch_size_stat_compute= batch_size_stat_compute
        self.gradient_clip_val= kwargs.get("gradient_clip_val", 0)
        self.patience= kwargs.get("patience", 30)
        self.NUM_COMP= NUM_COMP
        pin_memory= True
        #
        self.train_dataloader= DataLoader(train_dataset, batch_size= self.train_batch_size, collate_fn=collate_fn, num_workers=self.train_num_workers, pin_memory=pin_memory)

        if val_dataset:
            self.val_dataloader= DataLoader(val_dataset, batch_size= self.val_batch_size, collate_fn= collate_fn, num_workers=self.val_num_workers)
        else: 
            self.val_dataloader= None

        self.active_data= ActiveLearningData(pool_dataset)

        # 
        self.meta_df= read_meta_df("./meta_data/x_df.csv")
        #
        self.current_iteration=0
        self.populations= load_population_data(populaiton_csv_path)
        
    def initialize_data(self):
        #initialize train data/loader
        print("initializing train data ...", end="\n")
        indices= self.active_data.get_random_pool_indices(self.initial_train_size)
        self.acquire(indices)
        self.fetch(indices)
        data_z_stats=self.update_dataset_transform()
        self.model.update_y_stats(data_z_stats["y_mean"], data_z_stats["y_std"])

    #
    def learn(self, max_iter):
        
        for it in range(max_iter):

            self.current_iteration+=1

            trainer= self.get_trainer()

            trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

            # search candidates
            # profiler= cProfile.Profile()
            # profiler.enable()
            candidate_batch= self.search_candidate()
            self.acquire(candidate_batch)
            file_ids= self.fetch(candidate_batch)

            data_z_stats= self.update_dataset_transform()
            #
            # update the y_stats of the model
            self.model.update_y_stats(data_z_stats["y_mean"], data_z_stats["y_std"])
            #save
            self.save_trainer_info(trainer, data_z_stats, file_ids)  
            # profiler.disable()
            # stats= pstats.Stats(profiler)
            # stats.dump_stats(f"logs/acquisition-{self.current_iteration}.prof")
    #
    def search_candidate(self):
        print("iter={} ... searching for best batch".format(self.current_iteration))
        candidate_batch= self.acquisition.get_candidate_batch(self.model, self.active_data)
        return candidate_batch
    #
    def acquire(self, candidate_batch):
         self.active_data.acquire(candidate_batch)
    #
    def fetch(self, candidate_batch):

        print(" Fetching data ...", end="\n")
        x, xt, y0, run_ids, indices= self.get_x_data(candidate_batch)

        # save_x_data(x, run_ids, indices, Path(self.params["dataPath"]).joinpath("x_train"))
        save_x_data(x, run_ids, Path(self.dataPath).joinpath("x_train"))
        save_xt_data(xt, run_ids, Path(self.dataPath).joinpath("xt_train"))
        save_y0_data(y0, run_ids, Path(self.dataPath).joinpath("y0_train"))
        
        self.dataRetriever.fetch(run_ids)
        run_id_list= chain(*run_ids)
        file_ids= list(set(map_runid_to_fname(ri) for ri in run_id_list))
        return file_ids
    #
    def update_dataset_transform(self):

        x_mean, x_std, y_mean, y_std= get_z_score_stat(self.dataPath, self.x_dim, self.xt_dim, self.y_dim, self.seq_len, self.NUM_COMP, self.populations, category="train", batch_size= self.batch_size_stat_compute)
        # x_transform, y_transform= get_z_score_transform(x_mean, x_std, y_mean, y_std)
        y_hosp_inc_mean, y_hosp_prev_mean, y_latent_inc_mean, y_latent_prev_mean= np.split(y_mean, self.NUM_COMP, axis=-1)
        y_hosp_inc_std, y_hosp_prev_std, y_latent_inc_std, y_latent_prev_std= np.split(y_std, self.NUM_COMP, axis=-1)
        x_transform= None
        
        y_hosp_inc_transform= ZNormalize(y_hosp_inc_mean, y_hosp_inc_std)
        y_hosp_prev_transform= ZNormalize(y_hosp_prev_mean, y_hosp_prev_std)
        y_latent_inc_transform= ZNormalize(y_latent_inc_mean, y_latent_inc_std)
        y_latent_prev_transform= ZNormalize(y_latent_prev_mean, y_latent_prev_std)

        self.train_dataset.update_dataset()
        self.train_dataset.update_transforms(x_transform, y_hosp_inc_transform, y_hosp_prev_transform, y_latent_inc_transform, y_latent_prev_transform)

        if self.val_dataset:
            self.val_dataset.update_transforms(x_transform, y_hosp_inc_transform, y_hosp_prev_transform, y_latent_inc_transform, y_latent_prev_transform)

        # self.model.update_y_stats(y_mean, y_std)
        data_z_stat=dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std= y_std)

        return data_z_stat

    def get_trainer(self):

        callbacks=[pl.callbacks.EarlyStopping(monitor="train_loss", mode="min", patience=self.patience)]
        
        loggers=[pl.loggers.TensorBoardLogger("./logs"), pl.loggers.CSVLogger("./logs")]
        #profiler= PyTorchProfiler(f"logs/torch_profile/torch_profile_{self.current_iteration}.prof")
        trainer= pl.Trainer(default_root_dir="./logs",
                            accelerator=self.device, 
                            max_epochs=self.max_epochs,
                            callbacks= callbacks,
                            logger= loggers,
                            gradient_clip_val=self.gradient_clip_val)
        return trainer

    def save_trainer_info(self, trainer:pl.Trainer, data_z_scores:dict, file_ids:list):

        Path("./logs/active_logs").mkdir(exist_ok=True, parents=True)
        Path("./logs/checkpoints").mkdir(exist_ok=True, parents=True)
        info= dict(data_size= len(self.active_data.dataset),
                train_size= self.active_data.train_size,
                train_indices= self.active_data.training_dataset.indices.tolist(),
                train_file_ids= file_ids,
                last_train_epoch_loss= trainer.logged_metrics["train_loss"].detach().cpu().item(),
                last_train_epoch_wmape= trainer.logged_metrics["train_wmape"].detach().cpu().item(),
                last_val_epoch_mae=trainer.logged_metrics["val_mae"].detach().cpu().item(),
                last_val_epoch_mse=trainer.logged_metrics["val_mse"].detach().cpu().item(),
                last_val_epoch_wmape=trainer.logged_metrics["val_wmape"].detach().cpu().item())
        with open(f"logs/active_logs/active_info_iter_{str(self.current_iteration).zfill(10)}.yaml", "w") as fp:
            yaml.dump(info, fp)

        x_stats_filename= f"logs/active_logs/x_stats_iter_{str(self.current_iteration).zfill(10)}.parquet"
        pd.DataFrame({"x_mean":data_z_scores["x_mean"], "x_std":data_z_scores["x_std"]}).to_parquet(x_stats_filename)

        y_stats_filename= f"logs/active_logs/y_stats_iter_{str(self.current_iteration).zfill(10)}.parquet"
        pd.DataFrame({"y_mean":data_z_scores["y_mean"], "y_std":data_z_scores["y_std"]}).to_parquet(y_stats_filename)

        save_checkpoint_path= f"logs/checkpoints/checkpoint_iter_{str(self.current_iteration).zfill(10)}.pt"
        trainer.save_checkpoint(save_checkpoint_path)
                
    def get_x_data(self, indices):
        """
        extract data from dataset based on indices

        @return:
            x_data: numpy array
            run_ids: numpy array
            indices:numpy array
        """
        if isinstance(indices, torch.Tensor):
            indices= indices.detach().cpu().numpy()

        indices=list(indices)

        run_ids= self.meta_df.iloc[indices]["run_ids"].tolist()
        
        x= construct_x(self.meta_df.iloc[indices], self.x_col_names, self.frac_pops_names, self.populations, POPULATION_SCALER=self.POPULATION_SCALER)
        xt= construct_temporal_features(self.meta_df.iloc[indices], self.seq_len, self.num_nodes)
        y0= construct_initial_conditions(self.meta_df.iloc[indices], self.populations, POPULATION_SCALER= self.POPULATION_SCALER)

        # x: [num_samples,#nodes, x_dim]
        return np.array(x, dtype=np.float32), np.array(xt, dtype=np.float32), np.array(y0, dtype=np.float32), run_ids, indices

    def load_model(self):
        checkpoint_path= sorted(Path("logs/checkpoints").glob("*.pt"))[-1]
        active_log_path= sorted(Path("logs/active_logs").glob("*.yaml"))[-1]
        print("loading:")
        print("\t", checkpoint_path.name)
        print("\t", active_log_path.name)
        active_data= load_active_data(active_log_path)
        train_indices= active_data["train_indices"]
        last_iter_file_ids= active_data["train_file_ids"]
        self.current_iteration= int(active_log_path.stem.split("_")[-1])
        self.model= load_model(self.model, checkpoint_path)
        remove_extra_train_files(self.dataPath, last_iter_file_ids)
        self.active_data.acquire(train_indices)
        
        _=self.update_dataset_transform()
