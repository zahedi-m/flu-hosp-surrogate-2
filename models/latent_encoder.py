import torch

import torch.distributions as dist

from typing import Tuple

class LatentEncoder(torch.nn.Module):
    def __init__(self,
                r_dim:int,
                z_dim:int)->None:
        super().__init__()

        z_enc_hidden= 64

        # self.fc= torch.nn.Linear(r_dim, z_enc_hidden)        
        self.mu_layer= torch.nn.Linear(r_dim, z_dim)
        self.var_layer= torch.nn.Linear(r_dim, z_dim)

    def forward(self, r:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
            r:[L, r_dim]
            @ return :
                mu_z: [L, z_dim]
                digma_z: [L, z_dim]
        """ 
        # h= torch.relu(self.fc(r))

        mu_z= self.mu_layer(r)

        var_z= .001+ .999* torch.nn.functional.softplus(self.var_layer(r))
    
        return mu_z, var_z
