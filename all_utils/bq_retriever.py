
import shutil
from pathlib import Path

from google.cloud import bigquery
from google.cloud import bigquery_storage

import pandas as pd

import numpy as np

import pickle

from typing import Literal



class BQRetriever:
    """
        Big Query data retriver 
    """
    def __init__(self, destPath, category:Literal["train", "val", "test"]):
        
        self.destPath= destPath

        if not isinstance(destPath, Path):
            self.destPath= Path(destPath)

        self.project_id="mobs-mehdi"
        self.bqclient= bigquery.Client(project= self.project_id)
        self.bqstorage_client= bigquery_storage.BigQueryReadClient()

        self.category= category

        self.col_names=["run_id", "starting_date", "last_day_sims", "seasonality_min", "R0", "fraction_susceptible"]

        self.destPath.mkdir(exist_ok=True, parents=True)  
        self.destPath.joinpath("tmp").mkdir(exist_ok=True, parents=True)

        self.savePath= self.destPath.joinpath("y_inc_"+self.category)
        self.savePath.mkdir(exist_ok=True, parents=True)
    
    def fetch(self, run_ids_grouped:np.ndarray, indices):

        run_ids_ravel= list(np.hstack(run_ids_grouped))

        query="""
                    SELECT * FROM `mobs-mehdi.Exp1.incidence` 
                    WHERE date > "2018-01-01" AND run_id IN {run_ids_fetch}
                    """.format(run_ids_fetch=tuple(run_ids_ravel))

        df= self.bqclient.query(query).result().to_dataframe(bqstorage_client=self.bqstorage_client, progress_bar_type='tqdm')     
        df.to_csv(self.destPath.joinpath("tmp/y_df.csv"))

        np.save(self.destPath.joinpath("tmp/run_ids_grouped.npy"), run_ids_grouped)
        np.save(self.destPath.joinpath("tmp/indices.npy"), indices)

        preprocess_and_save(self.destPath.joinpath("tmp/y_df.csv"), run_ids_grouped, indices, self.savePath)


def preprocess_and_save(filename, run_ids_grouped, indices, destPath):

    print("preprocessing data ...", end="\n")

    if not isinstance(destPath, Path):
        destPath= Path(destPath)
    
    df_= pd.read_csv(filename, parse_dates=["date"])

    for run_ids, idx in zip(run_ids_grouped, indices): 
        
        df= df_.loc[df_["run_id"].isin(run_ids)].copy()
        
        data={
            "date":df.date,
            "country_id":df["country_id"],
            "run_id":df["run_id"],
            "Suceptible":df.loc[:, df.columns.str.startswith("Suceptible")].sum(axis=1).values.astype(np.float32),
            "Latent": df.loc[:, df.columns.str.startswith("Latent")].sum(axis=1).values.astype(np.float32),
            "Infectious_symptomatic_travel": df.loc[:, df.columns.str.startswith("Infectious_symptomatic_travel")].sum(axis=1).values.astype(np.float32),
            "Infectious_symptomatic_nontravel":df.loc[:, df.columns.str.startswith("Infectious_symptomatic_nontravel")].sum(axis=1).values.astype(np.float32),
            "Infectious_asymptomatic": df.loc[:, df.columns.str.startswith("Infectious_asymptomatic")].sum(axis=1).values.astype(np.float32),
            "Recovered":df.loc[:, df.columns.str.startswith("Recovered")].sum(axis=1).values.astype(np.float32)}
       
        df_processed= pd.DataFrame(data=data)
        
        arr= get_array(df_processed)
        filename= str(idx)+"_y_inc"+'.npy'
        np.save(destPath.joinpath(filename), arr)

def get_array(df:pd.DataFrame)->np.ndarray:

    list_y1=[]
    list_y2=[]
   
    for _, g1 in df.groupby("run_id", sort=False):
        
        g1.sort_values(by="date", inplace=True)

        list_latent=[]
        list_recovered=[]

        for _, g2 in g1.groupby("country_id", sort=False):
            # g2.sort_values(by="date")
            list_latent.append(g2["Latent"].values)
            list_recovered.append(g2["Recovered"].values)
        #
        list_y1.append(np.stack(list_latent, axis=-1))
        list_y2.append(np.stack(list_recovered, axis=-1))

    # y:[L, #compartment, #samples, #nodes]
    y1= np.stack(list_y1, axis=-1)
    y2= np.stack(list_y2, axis=-1)
    
    y= np.concatenate([y1, y2], axis=1)
    
    # y: [#samples, L, comp*#nodes]
    return y.transpose(2, 0, 1).astype(np.float32)

def save_x_data(x, run_ids:np.ndarray, indices:np.ndarray, col_names, destPath):
    
    """
        x:[#num_nodes, x_dim]
    """
    if not isinstance(destPath, Path):
        destPath= Path(destPath)
    
    for idx, run_ids, xx in zip(indices, run_ids, x):

        num_repeat= len(run_ids)
        
        data= np.repeat(xx[np.newaxis, ...], num_repeat, axis=0)
        
        filename= str(idx)+"_x.npy"

        np.save(destPath.joinpath(filename), data)




