import torch
from torch import Tensor

import torch.distributions as dist

import numpy as np

import yaml
from typing import Union


# build multilyaers perceptron
def build_MLP(in_features, hidden_dims:list, out_features):

    layers=[]
    layers=[]

    input_dim= hidden_dims[0]
    for hdim in hidden_dims[1:]:
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(input_dim, hdim))
        layers.append(torch.nn.ReLU())
        input_dim= hdim
    layers.append(torch.nn.Linear(hidden_dims[-1], out_features))

    return torch.nn.Sequential(*layers)

def load_model(model, checkpoint_path):
    
    checkpoint= torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_active_data(active_data_path):
    with open(active_data_path, "r") as fp:
        active_data= yaml.load(fp, Loader=yaml.SafeLoader)

    return active_data


def crps_normal(mu, sigma, y_true):
    """
    Compute the Continuous Ranked Probability Score (CRPS) for a batch of univariate normal distributions.
    Args:
        mu: [batch_size, seq_len, y_dim]
        sigma: [batch_size, seq_len, y_dim]
        y_true: [batch_size, seq_len, y_dim]
    Returns:
        crps:[batch_size, seq_len, y_dim]
    """
    batch_size, seq_len, y_dim = mu.size()
    c1 = torch.sqrt(torch.tensor(2 / np.pi)).to(mu.device)
    c2 = torch.tensor(0.5).to(mu.device)
    term1 = ((mu - y_true) ** 2) / (2 * sigma ** 2)
    term2 = sigma ** 2 / 12 * (c1 - c2)
    crps = term1 + term2
    return crps


def crps_gaussian(y:Tensor, mu:Tensor, sigma:Tensor):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); y)
    Formula taken from Equation (5):
    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004
    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
    Parameters
    ----------
    y :
        The observation or set of observations.
    mu : 
        The mean of the forecast normal distribution
    sig : 
        The standard deviation of the forecast distribution
    
    Returns
    -------
    crps : 
        The CRPS of each observation y relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    """

    # standadized x
    sx = (y- mu) / (sigma+1e-6)
    
    #standard normal
    normal= dist.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

    # some precomputations to speed up the gradient
    pdf = normal.log_prob(sx).exp()
    cdf = normal.cdf(sx)

    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sigma * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    
    return crps

###
def crps_by_each_cdf(y_true:Tensor, yhat:Tensor):
    yhat_sort, _= torch.sort(yhat, dim=0)
    n_samples= yhat_sort.size(0)
    cdf_yhat= torch.cumsum(torch.ones_like(yhat_sort, device= yhat.device, dtype=torch.float32), dim=0)/n_samples

    crps=[]
    for i in range(y_true.size(0)):
        heaviside= (yhat_sort>= y_true[i, :, :].unsqueeze(0)).int()
        crps_per_sample= ((cdf_yhat - heaviside) ** 2).mean(dim=0)  
        crps.append(crps_per_sample)
    #
    return torch.stack(crps, dim=0)

def crps_cdf(y_true:Tensor, yhat:Tensor):
    yhat_sort, _= torch.sort(yhat, dim=0)
    n_samples= yhat_sort.size(0)
    cdf_yhat= torch.cumsum(torch.ones_like(yhat_sort, device= yhat.device, dtype=torch.float32), dim=0)/n_samples
    #
    y_true_sort, _= torch.sort(y_true, dim=0)
    cdf_y_true= torch.cumsum(torch.ones_like(y_true, device= y_true.device, dtype= torch.float32), dim=0)/y_true_sort.size(0)

    crps= ((cdf_yhat- cdf_y_true)**2.0).mean()
    return crps, cdf_yhat, yhat_sort, cdf_y_true, y_true_sort

