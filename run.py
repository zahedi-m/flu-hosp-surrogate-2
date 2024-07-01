
import os
# os.environ['TORCH_CUDA_MEMORY_CK'] = 'caching:0.5,pool:0.1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

import torch

from pathlib import Path

from models.stnp import STNP

from activeLearner.active_dataset import ActiveLearningData
from activeLearner.active_learner import ActiveLearner

from all_utils.make_data import  make_data

from all_utils.bucket_retriever import BucketRetriever

from all_utils.acquisitions import MeanStd, LatentInfoGain

from all_utils.data_utils import get_datasets, load_graph_data
from all_utils.read_config import read_config_file

#
# torch.set_num_threads(10)
device= "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Num threads={torch.get_num_threads()}")
    config= read_config_file("./config.yaml")
    # # fetch and save val-test set
    metaPath= Path(config["meta_data"]["metaPath"])

    # 
    # make_data(metaPath.joinpath("all_meta_df.csv"), config["data"]["initial_test_size"], config["meta_data"]["dataPath"], "test", config["data"]["x_col_names"], config["data"]["frac_pops_names"], config["data"]["initial_col_names"], config["data"]["output_compartments"], config["model"]["seq_len"], config["model"]["num_nodes"], save_filename="x_val_df.csv", num_workers= config["train"]["val_num_workers"], bucket_name= config["meta_data"]["bucket_name"], basin_map_csv_path=config["meta_data"]["basin_map_csv_path"], population_csv_path=config["meta_data"]["population_csv_path"], POPULATION_SCALER= int(config["model"]["POPULATION_SCALER"]))
    # 
    # make_data(metaPath.joinpath("x_val_df.csv"), config["data"]["initial_val_size"], config["meta_data"]["dataPath"], "val", config["data"]["x_col_names"], config["data"]["frac_pops_names"], config["data"]["initial_col_names"], config["data"]["output_compartments"], config["model"]["seq_len"], config["model"]["num_nodes"], save_filename="x_df.csv", num_workers= config["train"]["val_num_workers"], bucket_name= config["meta_data"]["bucket_name"], basin_map_csv_path=config["meta_data"]["basin_map_csv_path"], population_csv_path= config["meta_data"]["population_csv_path"], POPULATION_SCALER= int(config["model"]["POPULATION_SCALER"]))

    train_dataset, val_dataset, pool_dataset= get_datasets(config["meta_data"]["metaPath"], config["meta_data"]["dataPath"], config["meta_data"]["srcPath"], config["data"]["x_col_names"], config["data"]["frac_pops_names"], config["data"]["initial_col_names"], config["model"]["seq_len"], config["model"]["num_nodes"], config["meta_data"]["population_csv_path"], config["model"]["POPULATION_SCALER"])

    edge_index, edge_weight= load_graph_data(config["meta_data"]["metaPath"])    

    active_dataset= ActiveLearningData(train_dataset)

    model= STNP(x_dim=config["model"]["x_dim"], 
                xt_dim=config["model"]["xt_dim"],
                y_dim=config["model"]["y_dim"],
                z_dim=config["model"]["z_dim"],
                r_dim=config["model"]["r_dim"],
                seq_len=config["model"]["seq_len"],
                num_nodes=config["model"]["num_nodes"],
                in_channels=config["model"]["in_channels"],
                out_channels=config["model"]["out_channels"],
                embed_out_dim=config["model"]["embed_out_dim"],
                max_diffusion_step=config["model"]["max_diffusion_step"],
                encoder_num_rnn=config["model"]["encoder_num_rnn"],
                decoder_num_rnn=config["model"]["decoder_num_rnn"],
                decoder_hidden_dims=config["model"]["decoder_hidden_dims"],
                NUM_COMP= config["model"]["NUM_COMP"],
                context_percentage=config["model"]["context_percentage"],
                lr=config["train"]["lr"],
                lr_encoder=config["train"]["lr_encoder"],
                lr_decoder=config["train"]["lr_decoder"],
                lr_milestones=config["train"]["lr_milestones"],
                lr_gamma=config["train"]["lr_gamma"],
                edge_index=edge_index, 
                edge_weight=edge_weight)

    acquisition= MeanStd(config["model"]["y_dim"], config["mstd"]["acquisition_size"], config["mstd"]["pool_loader_batch_size"], config["mstd"]["acquisition_pool_fraction"], device=device)
    # acquisition= LatentInfoGain(config["lig"]["acquisition_size"], config["lig"]["pool_loader_batch_size"], config["lig"]["acquisition_pool_fraction"], config["active_learner"]["pool_num_workers"], device= device)
    data_retriever= BucketRetriever(config["meta_data"]["dataPath"], category="train", compartments=config["data"]["output_compartments"], num_workers=config["active_learner"]["retriever_num_workers"], bucket_name=config["meta_data"]["bucket_name"], basin_map_csv_path=config["meta_data"]["basin_map_csv_path"])

    learner= ActiveLearner(model, 
                        x_dim= config["model"]["x_dim"],
                        xt_dim= config["model"]["xt_dim"],
                        y_dim= config["model"]["y_dim"],
                        seq_len=config["model"]["seq_len"],
                        num_nodes= config["model"]["num_nodes"], 
                        train_dataset=train_dataset, 
                        val_dataset= val_dataset, 
                        pool_dataset= pool_dataset, 
                        acquisition= acquisition, 
                        dataRetriever=data_retriever, 
                        max_epochs=config["train"]["max_epochs"],
                        train_batch_size= config["train"]["train_batch_size"],
                        train_num_workers=config["train"]["train_num_workers"],
                        initial_train_size= config["active_learner"]["initial_train_size"],
                        val_batch_size=config["train"]["val_batch_size"],
                        val_num_workers= config["train"]["val_num_workers"],
                        lr= config["train"]["lr"],
                        dataPath= config["meta_data"]["dataPath"],
                        x_col_names= config["data"]["x_col_names"],
                        frac_pops_names= config["data"]["frac_pops_names"],
                        device=device, 
                        pool_num_num_workers=config["active_learner"]["pool_num_workers"],
                        populaiton_csv_path=config["meta_data"]["population_csv_path"], 
                        POPULATION_SCALER= config["model"]["POPULATION_SCALER"], 
                        batch_size_stat_compute=config["active_learner"]["batch_size_stat_compute"],
                        lr_encoder=config["train"]["lr_encoder"],
                        lr_decoder=config["train"]["lr_decoder"], 
                        lr_gamma=config["train"]["lr_gamma"],
                        NUM_COMP= config["model"]["NUM_COMP"])

    learner.initialize_data()
    # learner.load_model()

    learner.learn(config["active_learner"]["max_iter"])


if __name__== "__main__":
    # import cProfile, pstats
    # profiler= cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats= pstats.Stats(profiler)
    # stats.dump_stats("logs/main-lig.prof")
