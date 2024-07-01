import pandas as pd
import yaml

from pathlib import Path


with open("config.yaml", "rb") as fp:
    params= yaml.load(fp, Loader=yaml.SafeLoader)

filename= Path(params["metaPath"]).joinpath("exp1-meta.csv.gz")
df= pd.read_csv(filename, parse_dates=["starting_date", "last_day_sims"])

col_names=["run_id", "starting_date", "last_day_sims", "seasonality_min", "R0", "fraction_susceptible"]

df= df[col_names]

df["fraction_susceptible"]=df["fraction_susceptible"].apply(eval)
df["fraction_susceptible"]=df["fraction_susceptible"].apply(lambda x: x[0])

col_names=df.columns.to_list()
col_names.remove("run_id")

groups= df.groupby(col_names)

id2realizations= {name: list(set(map(lambda x: x, group["run_id"].to_list()))) for name, group in groups}

params_dict=dict(zip(col_names, zip(*id2realizations.keys())))

df_grouped= pd.DataFrame({**params_dict, "run_ids":list(id2realizations.values())})


df_grouped.to_csv(Path(params["metaPath"]).joinpath("all_meta_x_df.csv"))


df= pd.read_csv(Path(params["metaPath"]).joinpath("all_meta_x_df.csv"))


df["run_ids"]=df["run_ids"].apply(eval)

print(df.head()) 