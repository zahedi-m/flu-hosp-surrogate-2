import torch
import torch.nn as nn

import torch.distributions as dist


from all_utils.torch_utils import build_MLP

class EncoderRNN(torch.nn.Module):
    def __init__(self,
                enc_in_dim:int,
                r_dim:int,
                num_rnn:int):
        
        super().__init__()

        self.num_rnn= num_rnn
        self.r_dim= r_dim

        self.rnn= torch.nn.GRU(enc_in_dim, r_dim, num_rnn, batch_first=True)
        
    def forward(self, enc_in)->torch.Tensor:
        """
            @input:
               enc_in:[B, L, enc_in_dim]
            
            @output:
                ri:[B, L, r_dim]
        """
        batch_size= enc_in.size(0)
        
        device= enc_in.device
        h0= torch.zeros((self.num_rnn, batch_size, self.r_dim), device= device)
        output, hn= self.rnn(enc_in, h0)

        return output

class DecoderRNN(torch.nn.Module):

    def __init__(self,
                embed_out_dim:int,
                z_dim:int, 
                hidden_dims:list,
                y_dim:int,
                num_rnn:int=1,
                **kwarg):

        super().__init__()

        self.hidden_dims= hidden_dims
        self.y_dim= y_dim
        self.num_rnn= num_rnn

        input_dim= embed_out_dim+ z_dim+ y_dim
        self.h0= nn.Parameter(torch.zeros(num_rnn, 1, hidden_dims[0]))

        self.rnn= nn.GRU(input_dim, hidden_dims[0], num_rnn, batch_first=True)
         
        fcs=[]
        input_dim= hidden_dims[0]
        for hdim in hidden_dims[1:]:
            fcs.append(nn.Linear(input_dim, hdim))
            fcs.append(nn.ReLU())
            input_dim= hdim

        self.fc_layers= nn.Sequential(*fcs)

        self.fc= nn.Linear(hidden_dims[-1], y_dim)

        self.softplus= nn.Softplus()
        self.relu= nn.ReLU()

    def forward(self, y0:torch.Tensor, embed_out:torch.Tensor, zs:torch.Tensor)->torch.Tensor:
        """
        @input:
            embed_out:[B, L, out_dim]
            zs: [B, L, z_dim]
            y0:[B, y_dim]
        @return:
            y_seq:[B, L, y_dim]
            
        """
        batch_size, seq_len, _= embed_out.size()
        h0= torch.zeros((self.num_rnn, batch_size, self.hidden_dims[0]), device= embed_out.device)
        # h0= self.h0.expand(-1, batch_size, -1).contiguous()
        #[B, 1, _]
        y_t= y0.unsqueeze(1)
        y_seq=[]
        
        for t in range(seq_len):

            inp= torch.cat([embed_out[:, t:t+1, ...], zs[:, t:t+1, ...], y_t], dim=-1)
            output, h0= self.rnn(inp, h0)
            output= self.relu(output)
            output= self.fc_layers(output)
            y_t=self.fc(output)
            y_seq.append(y_t)
        #
        y_seq= torch.cat(y_seq, dim=1)
        
        return y_seq


class DecoderRNN_2(torch.nn.Module):

    def __init__(self,
                embed_out_dim:int,
                z_dim:int, 
                hidden_dims:list,
                y_dim:int,
                init_val_hidden,
                init_val_pdrop,
                num_rnn:int=1,
                **kwarg):

        super().__init__()

        self.hidden_dims= hidden_dims
        self.y_dim= y_dim
        self.num_rnn= num_rnn

        input_dim= embed_out_dim+ z_dim+ y_dim

        self.rnn= nn.GRU(input_dim, hidden_dims[0], num_rnn, batch_first=True)
        self.h0= nn.Parameter(torch.zeros(num_rnn, 1, hidden_dims[0]))
        fcs=[]
        input_dim= hidden_dims[0]
        for hdim in hidden_dims[1:]:
            fcs.append(nn.Linear(input_dim, hdim))
            fcs.append(nn.ReLU())
            input_dim= hdim

        self.fc_layers= nn.Sequential(*fcs)

        self.fc_init_val=nn.Sequential(nn.Linear(y_dim, init_val_hidden), nn.ReLU(), nn.Dropout(init_val_pdrop), nn.Linear(init_val_hidden, y_dim))

        self.fc= nn.Linear(hidden_dims[-1], y_dim)

        self.softplus= nn.Softplus()
        self.relu= nn.ReLU()

    def forward(self, y0:torch.Tensor, embed_out:torch.Tensor, zs:torch.Tensor)->torch.Tensor:
        """
        @input:
            embed_out:[B, L, out_dim]
            zs: [B, L, z_dim]
            y0:[B, y_dim]
        @return:
            y_seq:[B, L, y_dim]
            
        """
        batch_size, seq_len, _= embed_out.size()
        # h0= torch.zeros((self.num_rnn, batch_size, self.hidden_dims[0]), device= embed_out.device)
        h0= self.h0.expand(-1, batch_size, -1).contiguous()
        #[B, 1, _]
        y_t= self.fc_init_val(y0).unsqueeze(1)
        y_seq=[]
        for t in range(seq_len):
            inp= torch.cat([embed_out[:, t:t+1, ...], zs[:, t:t+1, ...], y_t], dim=-1)
            output, h0= self.rnn(inp, h0)
            output= self.relu(output)
            output= self.fc_layers(output)
            y_t= self.fc(output)
            y_seq.append(y_t)
        #
        y_seq= torch.cat(y_seq, dim=1)
        return y_seq


class DecoderRNN_3(torch.nn.Module):

    def __init__(self,
                embed_out_dim:int,
                z_dim:int, 
                hidden_dims:list,
                y_dim:int,
                num_rnn:int=1,
                **kwarg):

        super().__init__()

        self.hidden_dims= hidden_dims
        self.y_dim= y_dim
        self.num_rnn= num_rnn

        input_dim= embed_out_dim+ z_dim

        self.rnn= nn.GRU(input_dim, hidden_dims[0], num_rnn, batch_first=True)
        self.h0= nn.Parameter(torch.zeros(num_rnn, 1, hidden_dims[0]))
        fcs=[]
        input_dim= hidden_dims[0]
        for hdim in hidden_dims[1:]:
            fcs.append(nn.Linear(input_dim, hdim))
            fcs.append(nn.ReLU())
            input_dim= hdim
        fcs.append(nn.Linear(hidden_dims[-1], y_dim))
        self.fc_layers= nn.Sequential(*fcs)


class DecoderRNN_4(torch.nn.Module):

    def __init__(self,
                embed_out_dim:int,
                z_dim:int, 
                hidden_dims:list,
                y_dim:int,
                num_rnn:int=1,
                **kwarg):

        super().__init__()

        self.hidden_dims= hidden_dims
        self.y_dim= y_dim
        self.num_rnn= num_rnn

        input_dim= embed_out_dim+ z_dim

        self.rnn= nn.GRU(input_dim, hidden_dims[0], num_rnn, batch_first=True)
        self.h0= nn.Parameter(torch.zeros(num_rnn, 1, hidden_dims[0]))
        fcs=[]
        input_dim= hidden_dims[0]
        for hdim in hidden_dims[1:]:
            fcs.append(nn.Linear(input_dim, hdim))
            fcs.append(nn.ReLU())
            input_dim= hdim
        self.fc_layers= nn.Sequential(*fcs)

        self.mu_f= nn.Linear(hidden_dims[-1], y_dim)
        self.var_fc= nn.Linear(hidden_dims[-1], y_dim)

        self.softplus= nn.Softplus()
        self.relu= nn.ReLU()

    def forward(self, y0:torch.Tensor, embed_out:torch.Tensor, zs:torch.Tensor)->torch.Tensor:
        """
        @input:
            embed_out:[B, L, out_dim]
            zs: [B, L, z_dim]
            y0:[B, y_dim]
        @return:
            y_seq:[B, L, y_dim]
            
        """
        batch_size, _, _= embed_out.size()
        h0= self.h0.expand(-1, batch_size, -1).contiguous().to(embed_out.device)
        
        inp= torch.cat([embed_out, zs], dim=-1)
        # output:[B, L, hidden]
        output, _= self.rnn(inp, h0)
        
        h_t= self.fc_layers(output)

        mu_t= self.mu_fc(h_t)
        var_t= self.softplus(self.var_fc(h_t))
        return mu_t, var_t
