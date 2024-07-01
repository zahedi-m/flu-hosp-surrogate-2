import pandas as pd
import numpy as np

import datetime


def construct_x(df_, x_col_names, frac_pops_names, populations, POPULATION_SCALER=1.0):
    """  
      @return:
        [B, 51, x_dim+#frac_prev_pop] 
    """
    pops= populations/float(POPULATION_SCALER)
    df= df_.copy()

    df["days"]/=366
    x= df[x_col_names].values
    if np.ndim(x)<2:
        x= x[np.newaxis,...]

    x_frac_pops= df[frac_pops_names].values

    if x_frac_pops.ndim<2:
        x_frac_pops= x_frac_pops[np.newaxis,...]
    
    #[B, 51, #comp]
    
    # x_pops= np.einsum("ij,k->ikj", x_frac_pops.astype(np.float64), pops)
    x_frac_pops= np.repeat(x_frac_pops[:, np.newaxis, :], len(pops),axis=1)
   
    x= np.repeat(x[:, np.newaxis, :], len(pops),axis=1)
    
    # population feature
    pop_feat= pops[np.newaxis, :, np.newaxis]
    pop_feat= np.repeat(pop_feat, x_frac_pops.shape[0], axis=0)
    
    return np.concatenate([ x, x_frac_pops, pop_feat], axis=-1)

def construct_initial_conditions(df, populations, POPULATION_SCALER=1.0):
    latent= (df[["Latent"]].values+ df[["Latent_vax"]].values).astype(np.float64)
    if latent.ndim>1:
        if np.squeeze(latent).ndim==0:
            latent= np.squeeze(latent).reshape(1)
        else:
            latent= np.squeeze(latent)
    y0_latent= np.einsum("i,j-> ij",latent, populations)
    return np.round(y0_latent)

#
def construct_temporal_features(df, seq_len, num_nodes):
    
    if isinstance(df[["starting_date", "seasonality_min"]], pd.Series):
        values= [df[["starting_date", "seasonality_min"]].to_list()]
    else:
        values= df[["starting_date", "seasonality_min"]].values.tolist()

    seasonality=  np.squeeze(np.stack([get_seasonality(x[0], x[1], seq_len) for x in values]))

    # [B, L, #nodes, xt_dim]
    seasonality=seasonality[..., np.newaxis, np.newaxis]
    seasonality= np.repeat(seasonality, num_nodes, axis=-2)

    return seasonality
    
def get_seasonality(starting_date, seasonality_min, seq_len):

    if isinstance(starting_date, pd.Timestamp):
        starting_date= starting_date.to_pydatetime()

    one_day= datetime.timedelta(days=1)
    dates= [starting_date+ t* one_day for t in range(seq_len+1)]
 
    seasonality=np.squeeze(np.array([apply_seasonality(day, seasonality_min, seasonality_max=1.0) for day in dates]))
    return seasonality

def get_starting_date_array(df_starting_date):
    df= df_starting_date.copy()
    
    df= pd.to_datetime(df)

    if isinstance(df, pd.Timestamp):
        year, month, day= pd.to_numeric(df.year), pd.to_numeric(df.month), pd.to_numeric(df.day)
    elif isinstance(df, pd.Series):
         year, month, day= pd.to_numeric(df.dt.year), pd.to_numeric(df.dt.month), pd.to_numeric(df.dt.day)
    else:
        raise TypeError
    
    year= year.values if np.ndim(year) !=0 else np.array([year])
    month= month.values if np.ndim(month) !=0 else np.array([month])
    day= day.values if np.ndim(day) !=0 else np.array([day])

    return np.stack([year, month, day], axis=-1)

def get_date_features_from_numpy(arr, t):
    B, nodes, _= arr.shape

    one_step= datetime.timedelta(days=1)

    # date_list=[datetime.datetime(*x)+ t * one_step for x in arr]
    date_list= convert_array_to_datetime(arr)

    dates=[np.array([d.year%100, d.month, d.day]) for d in date_list]

    dates= np.stack(dates, axis=0)[np.newaxis, ...]
    dates= np.repeat(dates, B, axis=0)

    return dates

def convert_array_to_datetime(arr):
    # all starting dates are the same
    x= arr[0].astype(np.int64)
    dates=[datetime.datetime(*x[i,:]) for i in range(x.shape[0])]
    return dates

def apply_seasonality(day, seasonality_min, seasonality_max=1.0):
    s_r = seasonality_min/seasonality_max
    day_max_north = datetime.datetime(day.year,1,15) # this computes Jan 15th for the year of the "day"     
    # north hemisphere
    seasonal_adjustment = 0.5*((1-s_r)*np.sin(2*np.pi/365*((day-day_max_north).days) + 0.5*np.pi)+1+s_r)
    return seasonal_adjustment
