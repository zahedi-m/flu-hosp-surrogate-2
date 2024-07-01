import pandas as pd
def run_gleam(df_input:pd.DataFrame)->list:
    """
        @input:
            df_input: size [query_size, num_cols]
            columns:
                R0: float
                seasonality_min: float
                starting_date: str
                Susceptible:float, 
                Latent: float
                Recovered:float
                ? population:float (if we have different parameters for each state)
    """
    filenames= None
    return filenames