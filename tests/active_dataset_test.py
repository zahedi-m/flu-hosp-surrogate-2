
from activeLearner.active_dataset import ActiveLearningData
from all_utils.data_utils import get_datasets
from all_utils.read_config import read_config_file
from torch.utils.data import DataLoader

import numpy as np

def test():
    config= read_config_file("config.yaml")
    train_dataset, val_dataset, pool_dataset= get_datasets(config["meta_data"]["metaPath"], config["meta_data"]["dataPath"], config["meta_data"]["srcPath"], config["data"]["x_col_names"], config["data"]["frac_pops_names"], config["data"]["initial_col_names"], config["model"]["seq_len"], config["model"]["num_nodes"], config["meta_data"]["population_csv_path"], config["model"]["POPULATION_SCALER"])
    active_dataset= ActiveLearningData(pool_dataset)
    
   
    print(len(train_dataset))
    print(len(active_dataset.pool_dataset))
    print(len(active_dataset.training_dataset))
    print(active_dataset.train_size)
    loader= DataLoader(active_dataset.pool_dataset, shuffle=False, batch_size=1)
    before_select= next(iter(loader))
    print("before select-->", before_select[-1])
    selected_indices= [i for i in range(10)]

    print("----")
    print(selected_indices)
    indices= active_dataset.get_dataset_indices(selected_indices)
    print(indices)
    # print(active_dataset.pool_dataset[indices[0]])
    active_dataset.acquire(selected_indices)
    # print(active_dataset.pool_dataset[indices[0]])

    print(len(active_dataset.training_dataset))
    print(len(active_dataset.pool_dataset))

    loader= DataLoader(active_dataset.pool_dataset, shuffle=False, batch_size=1)
    after_select=next(iter(loader))

    print("after select", after_select[-1])


          
if __name__=="__main__":
    test()