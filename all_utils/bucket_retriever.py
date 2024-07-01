import pandas as pd

from typing import Literal, Tuple

from pathlib import Path

import numpy as np

from google.cloud import storage

from multiprocessing.pool import ThreadPool

from itertools import chain
from functools import partial

import subprocess

MAX_LENGTH=245
 

class BucketRetriever:

    def __init__(self, destPath, 
                category:Literal["train", "val", "test"], 
                compartments:list, #["Latent", "Hospitalized"]
                num_workers:int,
                bucket_name, 
                basin_map_csv_path):
        
        #["Latent", "Hospitalized"]
        self.destPath= Path(destPath) if not isinstance(destPath, Path) else destPath

        self.category= category
        self.compartments=  compartments
        self.df_basin= pd.read_csv(basin_map_csv_path)
        self.destPath.mkdir(exist_ok=True, parents=True)  
        self.bucket_name= bucket_name
        self.num_workers= num_workers
        
        # temporary storage
        self.tmp= self.destPath.joinpath("tmp")
        self.tmp.mkdir(exist_ok=True, parents=True)
        [file.unlink(file) for file in self.tmp.iterdir()]
        
        #inicdence
        self.inc_savePath= self.destPath.joinpath("y_inc_"+ category)
        self.inc_savePath.mkdir(exist_ok=True, parents=True)
        
        #prevalence
        self.prev_savePath= self.destPath.joinpath("y_prev_"+ category)
        self.prev_savePath.mkdir(exist_ok=True, parents=True)

        storage_client= storage.Client()
        self.bucket= storage_client.bucket(bucket_name)

    # def fetch(self, run_ids:list, indices:list):
    def fetch(self, run_ids:list):
        """ Sequential implemention """
        
        print("Querying and preprocessing data ...", end="\n")
        
        run_id_list= list(chain(*run_ids))
        # 
        inc_filenames=list(set(get_incidence_fname(rid) for rid in run_id_list))
        inc_filenames= [f"gs://{self.bucket_name}/{f}" for f in inc_filenames]
        prev_filenames=list(set(get_prevalence_fname(rid) for rid in run_id_list))
        prev_filenames= [f"gs://{self.bucket_name}/{f}" for f in prev_filenames]
        download_bucket_files(self.tmp, inc_filenames+prev_filenames)

        with ThreadPool(processes=self.num_workers) as pool:
            pool.map(self.task, self.tmp.glob("*.gz"))
            pool.close()
            pool.join()

        print("\tdeleting temp files...")
        [file.unlink(file) for file in self.tmp.iterdir()]
        #

    def task(self, filename):    
        preprocess_and_save(filename, self.df_basin, self.destPath, compartments=self.compartments , category=self.category)
   
def preprocess_and_save(filename:Path, df_basin, destPath:Path, compartments, category):

    other_cols=["date", "state_name", "run_id", "fraction", "basin_id"]
    
    if filename.stem.split("_")[3]=="incidence":
        y_filename_suffix="y_inc"
        save_path= destPath.joinpath("y_inc_"+ category)
    elif filename.stem.split("_")[3]=="prevalence":
        y_filename_suffix= "y_prev"
        save_path= destPath.joinpath("y_prev_"+ category)
    else:
        raise FileExistsError
    
    df= pd.read_csv(filename, parse_dates=["date"])
    # sum over age groups
    for comp in compartments:
        target_cols=df.columns[df.columns.str.startswith(comp)].to_list()
        df[comp]= df[target_cols].sum(axis=1)

    df=df.merge(df_basin, how="inner", on="basin_id")
    df[compartments]= df[compartments].multiply(df.fraction, axis="index")
    df= df.groupby(["date", "run_id", "state_name"]).sum().reset_index()

    # make array
    df= df.sort_values(by=["state_name", "date", "run_id"])
    df_pivot= df.pivot_table(index=["run_id", "date"], columns=["state_name"], values=compartments, aggfunc='first')
    
    n_run_id= len(df["run_id"].unique())
    n_comps= len(compartments)
    n_states= len(df["state_name"].unique())
    arr= df_pivot[compartments].values.reshape(n_run_id, -1, n_comps*n_states)
    y_filename= make_y_filename(filename, y_filename_suffix)
    np.save(save_path.joinpath(y_filename), arr)
#
def make_y_filename(filename, y_filename_suffix):
    return filename.stem.split('_')[1]+f"_{y_filename_suffix}.npy"

def get_array_2(df, compartments):

    state_names= sorted(df["state_name"].unique())
    rids= sorted(df["run_id"].unique())
    
    df_grouped= df.groupby(["run_id", "state_name", "date"], sort=True).agg({comp:np.sum for comp in compartments})

    comps_list=[]
    for comp in compartments:
        comp_series= df_grouped[comp]

        a=np.stack([np.stack([comp_series.loc[(rid, state_name)] for state_name in  state_names], axis=1) for rid in rids])
        comps_list.append(a)
    #
    return  np.concatenate(comps_list, axis=-1)
#
def download_bucket_files(save_path, gcs_filenames:list): 
    cmd= f"gsutil -m cp {' '.join(gcs_filenames)} {save_path}"
    subprocess.run(cmd.split(), stdout= subprocess.PIPE, stderr=subprocess.PIPE, text=True)


##
def get_bucket_files(save_path, bucket, gcs_filenames): 
    
    if not isinstance(save_path, Path):
        save_path= Path(save_path)

    for index, gcs_filename in gcs_filenames:
        blob= bucket.blob(gcs_filename)
        if blob.exists():
            filename= f"{index}_"+gcs_filename.split("/")[-1]
            blob.download_to_filename(save_path.joinpath(filename))

def get_incidence_fname(run_id):
    fnumber = map_runid_to_fname(run_id)
    return f'experiment-2/out{fnumber}_{fnumber}_basins_incidence_daily.csv.gz'

def get_prevalence_fname(run_id):
    fnumber = map_runid_to_fname(run_id)
    return f'experiment-2/out{fnumber}_{fnumber}_basins_prevalence_daily.csv.gz'

# 
def map_runid_to_fname(run_id, version_id=61001, run_ids_per_file=6):
  offset = version_id * (10**12)
  return (offset + ((run_id - offset -1)//run_ids_per_file) * run_ids_per_file )


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

def get_bucket_file(gcs_filename): 
    try:
        df= pd.read_csv(gcs_filename)
    except:
        df= None
    return df

# def preprocess_and_save(filename:Path, df_map, destPath, y_filename_suffix, compartments):

#     other_cols=["date", "state_name", "run_id", "fraction", "basin_id"]

#     if not isinstance(destPath, Path):
#         destPath= Path(destPath)
    
#     df= pd.read_csv(filename, parse_dates=["date"])
#     # sum over age groups
#     for comp in compartments:
#         target_cols=df.columns[df.columns.str.startswith(comp)].to_list()
#         df[comp]= df[target_cols].sum(axis=1)

#     df=df.merge(df_map, how="inner", on="basin_id")

#     df[compartments]= df[compartments].multiply(df.fraction, axis="index")

#     # make array
#     df= df.sort_values(by=["state_name", "date", "run_id"])
#     df_pivot= df.pivot_table(index=["run_id", "date"], columns=["state_name"], values=compartments, aggfunc='first')
    
#     n_run_id= len(df["run_id"].unique())
#     n_comps= len(compartments)
#     n_states= len(df["state_name"].unique())
#     arr= df_pivot[compartments].values.reshape(n_run_id, -1, n_comps*n_states)
#     y_filename= make_y_filename(filename, y_filename_suffix)
#     np.save(destPath.joinpath(y_filename), arr)
