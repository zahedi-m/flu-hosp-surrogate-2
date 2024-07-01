#
from pathlib import Path
import numpy as np
from all_utils.bucket_retriever import map_runid_to_fname

def save_x_data(x, run_ids:np.ndarray, destPath):
    
    """
        x:[#num_nodes, x_dim]
    """
    
    if not isinstance(destPath, Path):
        destPath= Path(destPath)
    destPath.mkdir(exist_ok=True, parents=False)
    for run_ids, xx in zip(run_ids, x):
        file_id = map_runid_to_fname(run_ids[0])
        num_repeat= len(run_ids)
        data= np.repeat(xx[np.newaxis, ...], num_repeat, axis=0)
        filename= f"{file_id}_x.npy"
        np.save(destPath.joinpath(filename), data)


# def save_x_data(x, run_ids:np.ndarray, indices:np.ndarray, destPath):
    
#     """
#         x:[#num_nodes, x_dim]
#     """
    
#     if not isinstance(destPath, Path):
#         destPath= Path(destPath)
    
#     destPath.mkdir(exist_ok=True, parents=False)
    
#     for idx, run_ids, xx in zip(indices, run_ids, x):

#         num_repeat= len(run_ids)
        
#         data= np.repeat(xx[np.newaxis, ...], num_repeat, axis=0)
        
#         filename= str(idx).zfill(10)+"_x.npy"

#         np.save(destPath.joinpath(filename), data)

def save_xt_data(xt, run_ids:np.ndarray, destPath):
    
    """
        xt:[B, L, #num_nodes, xt_dim]
    """
    
    if not isinstance(destPath, Path):
        destPath= Path(destPath)
    
    destPath.mkdir(exist_ok=True, parents=False)
    
    for rid, xx in zip(run_ids, xt):
        file_id= map_runid_to_fname(rid[0])
        num_repeat= len(rid)
        data= np.repeat(xx[np.newaxis, ...], num_repeat, axis=0)
        # filename= str(idx).zfill(10)+"_xt.npy"
        filename= f"{file_id}_xt.npy"
        np.save(destPath.joinpath(filename), data)
#

def save_y0_data(y0, run_ids:np.ndarray, destPath):
    
    """
        y0:[y_dim]
    """
    
    if not isinstance(destPath, Path):
        destPath= Path(destPath)
    
    destPath.mkdir(exist_ok=True, parents=False)
    
    for rid, yy in zip(run_ids, y0):
        file_id= map_runid_to_fname(rid[0])
        num_repeat= len(rid)
        data= np.repeat(yy[np.newaxis, ...], num_repeat, axis=0)
        filename= f"{file_id}_y0.npy"
        np.save(destPath.joinpath(filename), data)

def make_x_filename(rid):
    if isinstance(rid, list):
        ri= rid[0] 
    elif isinstance(rid, str) or isinstance(rid, int):
        rid= int(rid)
    return f"{map_runid_to_fname(ri)}_x.npy"

def make_xt_filename(rid):
    if isinstance(rid, list):
        ri= rid[0] 
    elif isinstance(rid, str) or isinstance(rid, int):
        rid= int(rid)
    return f"{map_runid_to_fname(ri)}_xt.npy"

def make_y_filename(rid, y_filename_suffix):
    if isinstance(rid, list):
        ri= rid[0] 
    elif isinstance(rid, str) or isinstance(rid, int):
        rid= int(rid)
    return f"{map_runid_to_fname(ri)}_{y_filename_suffix}.npy"

def make_y0_filename(rid):
    if isinstance(rid, list):
        ri= rid[0] 
    elif isinstance(rid, str) or isinstance(rid, int):
        rid= int(rid)
    return f"{map_runid_to_fname(ri)}_y0.npy"