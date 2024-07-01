import pandas as pd
import numpy as np

def consistent_run_id(x):
    if len(x)>1:
        return [x[np.random.choice(len(x))]]
    else:
        return x


def make_unique():
    df= pd.read_csv("./meta_data/original_meta_df.csv")


    df["run_ids"]= df["run_ids"].apply(eval)

    df["run_ids"]= df["run_ids"].apply(lambda x: consistent_run_id(x))

    df.to_csv("./meta_data/all_meta_df.csv", index=False)

if __name__== "__main__":
    make_unique()