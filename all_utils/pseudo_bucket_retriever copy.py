import pandas as pd

from typing import Literal, Tuple

from pathlib import Path

import numpy as np

import shutil

from multiprocessing.pool import ThreadPool

MAX_LENGTH=245

class PseudoBucketRetriever:

    def __init__(self, srcPath, destPath, category:Literal["train", "val", "test"]):
        
        self.df_map= pd.read_csv("meta_data/mapping_basin_to_states_GLEAM.csv")

        self.srcPath= Path(srcPath) if not isinstance(srcPath, Path) else srcPath
        self.destPath= Path(destPath) if not isinstance(destPath, Path) else destPath

        self.col_names=["run_id", "seasonality_min", "R0", "fraction_susceptible"]
        
        self.destPath.mkdir(exist_ok=True, parents=True)  
        self.destPath.joinpath("tmp").mkdir(exist_ok=True, parents=True)

        #inicdence
        self.inc_savePath= self.destPath.joinpath("y_inc_"+ category)
        self.inc_savePath.mkdir(exist_ok=True, parents=True)
        

        #prevalence
        self.prev_savePath= self.destPath.joinpath("y_prev_"+ category)
        self.prev_savePath.mkdir(exist_ok=True, parents=True)

    def fetch(self, run_ids:np.ndarray, indices:np.ndarray):
        """
            Multiprocessing implementation          
        """
        print("Querying and preprocessing data ...", end="\n")

        with ThreadPool(processes=8) as pool:
            pool.map(self.task, zip(run_ids, indices))
    
    def task(self, tp:Tuple):

        rids, index= tp

        # for rid in rids:
        #     incidence_gcs_fname= get_incidence_fname(rid)
        #     prevalence_gcs_fname= get_prevalence_fname(rid)

        #     inc_filename_=self.srcPath.joinpath(incidence_gcs_fname)  
        #     if inc_filename_.is_file():
        #         inc_filename= inc_filename_
                
        #     prev_filename_= self.srcPath.joinpath(prevalence_gcs_fname)
        #     if prev_filename_.is_file():
        #         prev_filename= prev_filename_

        # df_incidence= pd.read_csv(inc_filename)
        # df_prevalence= pd.read_csv(prev_filename)

        df_incidence=None
        df_prevalence= None
        
        for rid in rids:

            incidence_gcs_fname= self.srcPath.joinpath(get_incidence_fname(rid))
            prevalence_gcs_fname= self.srcPath.joinpath(get_prevalence_fname(rid))
        
            df_read= get_bucket_file(incidence_gcs_fname)

            if df_read is not None:
                if df_incidence is None:
                    df_incidence= df_read.copy()
                else:
                    df_incidence.append(df_read)
            else:
                continue

            #
            df_read= get_bucket_file(prevalence_gcs_fname)
            
            if df_read is not None:
                if df_prevalence is None:
                    df_prevalence= df_read.copy()
                else:
                    df_prevalence.append(df_read)
            else:
                continue

        #save incidence

        if df_incidence is None or df_prevalence is None:
            return 

        preprocess_and_save(df_incidence, self.df_map, index, self.inc_savePath, y_filename_suffix="y_inc")
        # save prevalence
        preprocess_and_save(df_prevalence, self.df_map, index, self.prev_savePath, y_filename_suffix="y_prev")
    ##
    # def fetch(self, run_ids:np.ndarray, indices:np.ndarray):
    #     """ Sequential implemention """
        
    #     print("Querying and preprocessing data ...", end="\n")

    #     for rids, index in zip(run_ids, indices):

    #         for rid in rids:
    #             incidence_gcs_fname= get_incidence_fname(rid)
    #             prevalence_gcs_fname= get_prevalence_fname(rid)
            
    #             inc_filename=self.srcPath.joinpath(incidence_gcs_fname)

    #             if inc_filename.is_file():
    #                 df_incidence= pd.read_csv(inc_filename)
    #                 print(inc_filename, end="\n")

    #             prev_filename= self.srcPath.joinpath(prevalence_gcs_fname)
    #             if prev_filename.is_file():
    #                 df_prevalence= pd.read_csv(self.srcPath.joinpath(prevalence_gcs_fname))
    #                 print(prev_filename, end="\n\n")

    #         #save incidence
    #         preprocess_and_save(df_incidence, index, self.inc_savePath, y_filename_suffix="y_inc")
   
    #         # save prevalence
    #         preprocess_and_save(df_prevalence, index, self.prev_savePath, y_filename_suffix="y_prev")

def get_bucket_file(gcs_filename): 
    try:
        df= pd.read_csv(gcs_filename)
    except:
        df= None
    return df

def get_incidence_fname(run_id):
    fnumber = run_id
    return f'experiment-2/out{fnumber}_{fnumber}_basins_incidence_daily.csv.gz'

def get_prevalence_fname(run_id):
    fnumber = run_id
    return f'experiment-2/out{fnumber}_{fnumber}_basins_prevalence_daily.csv.gz'

def preprocess_and_save(df, df_map, idx, destPath, y_filename_suffix):

    if not isinstance(destPath, Path):
        destPath= Path(destPath)
    
    comp_names=["Latent", "Hospitalized"]

    df_state= df_map.merge(df, how="left", on="basin_id")
    
    df_state= pd.DataFrame(get_data(df_state))
    
    arr= get_array(df_state, comp_names)

    y_filename= str(idx)+"_"+y_filename_suffix+".npy"

    np.save(destPath.joinpath(y_filename), arr)

def get_array(df, comp_names):

    comp_dict_array={k:[] for k in comp_names}
   
    for _, g1 in df.groupby("run_id", sort=False):
        
        comp_dict_by_state= {k:[] for k in comp_names}

        for state, g2 in g1.groupby("state_name", sort=False):
            for comp in comp_dict_by_state.keys():
                comp_dict_by_state[comp].append(g2.groupby("date", sort=True, group_keys=True).sum()[comp].values)
        #
        # combine states: [L, #states]
        for k in comp_dict_by_state.keys():
            comp_dict_array[k].append(np.stack(comp_dict_by_state[k], axis=-1))
        
    #
    #concat compartments->(samples, L, #num_comp* states)
    y= np.concatenate([v for v in comp_dict_array.values()], axis=-1)

    y= y[:, :MAX_LENGTH,...]
    return y

def get_data(df):
    data={
        "date": df["date"].values,
        "basin_id": df["basin_id"].values,
        "state_name":df["state_name"].values, 
        "run_id":df["run_id"].values,
        "Susceptible": df.loc[:, df.columns.str.startswith("Suceptible")].multiply(df["fraction"], axis="index").sum(axis=1).values,
        "Infectious": df.loc[:, df.columns.str.startswith("Infectious")].multiply(df["fraction"], axis="index").sum(axis=1).values,
        "Hospitalized": df.loc[:, df.columns.str.startswith("Hospitalized")].multiply(df["fraction"], axis="index").sum(axis=1).values,
        "Removed": df.loc[:, df.columns.str.startswith("Removed")].multiply(df["fraction"], axis="index").sum(axis=1).values,
        "Latent": df.loc[:, df.columns.str.startswith("Latent")].multiply(df["fraction"], axis="index").sum(axis=1).values,
        "Recovered": df.loc[:, df.columns.str.startswith("Recovered")].multiply(df["fraction"], axis="index").sum(axis=1).values
    }
    return data