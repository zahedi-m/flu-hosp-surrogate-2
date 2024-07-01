

import torch

from tgnn.dcrnn import DCRNN2

class EmbedModel(torch.nn.Module):

    def __init__(self, 
                in_channels:int,
                embed_out_dim:int, 
                out_channels:int, 
                max_diffusion_step:int,
                num_nodes:int,
                **kwarg):
        super().__init__()

        # self.date_embed_in_dim= kwarg["date_embed_in_dim"]
                
        self.recurrent= DCRNN2(in_channels, out_channels, max_diffusion_step)

        self.fc= torch.nn.Linear(out_channels*num_nodes, embed_out_dim)

        # self.fc_date_embed= torch.nn.Linear(kwarg["date_embed_in_dim"], kwarg["date_embed_out_dim"])
    
    def forward(self, inputs, edge_index, edge_weight, hidden_state=None):
        """
        @input:
            inputs: shape (batch_size, self.num_nodes, self.input_dim)
            edge_index [2, num_edges]
            edge_weight [2, num_edges]
            hidden_state: (num_layers, batch_size, num_nodes, out_channesl)
               
        @return: 
                output: [batch_size, embed_out_dim]
                hidden_state [num_layers, batch_size, num_nodes, channel_out]

        """
        batch_size= inputs.size(0)

        # date_input= inputs[..., :self.date_embed_in_dim]
        # _inputs= inputs[..., self.date_embed_in_dim:]    

        # date_embed_out= self.fc_date_embed(date_input)

        # inputs= torch.cat([date_embed_out, _inputs], dim=-1)

        hidden_states= self.recurrent(inputs, edge_index, edge_weight, hidden_state)
        output = self.fc(hidden_states.reshape(batch_size, -1)) #aggregating all states

        return output, hidden_states

