import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from all_utils.bucket_retriever import BucketRetriever
from all_utils.save_utils import  save_x_data, save_y0_data, save_xt_data
from all_utils.data_utils import  read_meta_df, load_population_data
from all_utils.input_constructors import construct_x, construct_initial_conditions, construct_temporal_features

def make_data(metaFile, 
              size:int, 
              destPath, 
              category:Literal["test", "val"], 
              x_col_names, 
              frac_pops_names, 
              initial_col_names, 
              output_compartments, 
              seq_len, 
              num_nodes, 
              save_filename,
              num_workers,
              bucket_name, 
              basin_map_csv_path, 
              population_csv_path,
              POPULATION_SCALER
              ):
    
    print(f"Creating {category} data ...")
    
    destPath= Path(destPath) if not isinstance(destPath, Path) else destPath
        
    x_path=destPath.joinpath("x_"+category)
    x_path.mkdir(exist_ok=False, parents=True)

    xt_path=destPath.joinpath("xt_"+category)
    xt_path.mkdir(exist_ok=False, parents=True)

    y0_path= destPath.joinpath("y0_"+category)
    y0_path.mkdir(exist_ok=False, parents= True)

    populations= load_population_data(population_csv_path)
    
    x, xt, y0, run_ids, indices= get_random_data(metaFile, size, x_col_names, frac_pops_names, initial_col_names, seq_len, num_nodes, save_filename, populations, POPULATION_SCALER)

    save_x_data(x, run_ids, x_path)
    save_xt_data(xt, run_ids, xt_path)
    
    save_y0_data(y0, run_ids, y0_path)

    retriever= BucketRetriever(destPath, category, compartments=output_compartments, num_workers= num_workers, bucket_name=bucket_name, basin_map_csv_path=basin_map_csv_path)
    
    retriever.fetch(run_ids)

def get_random_data(metaFile:Path, size, col_names, frac_pops_names, initial_col_names, seq_len, num_nodes, save_filename, populations, POPULATION_SCALER):

    parent_path= metaFile.parents[0]

    df= read_meta_df(metaFile)

    indices= np.random.choice(df.index.values, size, replace=False)  

    x= construct_x(df.iloc[indices], col_names, frac_pops_names, populations, POPULATION_SCALER)
    xt= construct_temporal_features(df.iloc[indices], seq_len, num_nodes)
    y0_df= df.iloc[indices][initial_col_names]
    y0= construct_initial_conditions(y0_df, populations, POPULATION_SCALER)

    run_ids= df.iloc[indices]["run_ids"].apply(lambda x: np.array(x)).values    
    
    df= df.drop(index=indices)
    df.to_csv(parent_path.joinpath(save_filename), index=False)
    
    return x, xt, y0, run_ids, indices
