import torch
from torch import Tensor
import torch.distributions as dist

import pytorch_lightning as pl

import numpy as np

from .embed import EmbedModel

from .latent_encoder import LatentEncoder

from .rnn_coder import DecoderRNN, EncoderRNN, DecoderRNN_3

from .aggregator import MeanAggregator

from typing import Tuple

from torchmetrics import WeightedMeanAbsolutePercentageError

# DEVICE= torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

class STNP(pl.LightningModule):

    def __init__(self, 
                x_dim, 
                xt_dim,
                y_dim,
                z_dim,
                r_dim,
                seq_len,
                num_nodes,
                in_channels,
                out_channels,
                embed_out_dim,
                max_diffusion_step,
                encoder_num_rnn,
                decoder_num_rnn,
                decoder_hidden_dims,
                NUM_COMP,
                context_percentage,
                lr,
                lr_encoder,
                lr_decoder, 
                lr_milestones,
                lr_gamma,
                edge_index, 
                edge_weight,
                ):
        super().__init__()

        # self.hparams.update(params)
        self.save_hyperparameters("x_dim", "xt_dim", "y_dim", "z_dim", "r_dim", "seq_len", "num_nodes", "in_channels", "out_channels", "embed_out_dim", "max_diffusion_step",
                "encoder_num_rnn", "decoder_num_rnn", "decoder_hidden_dims", "context_percentage", "lr", "lr_encoder", "lr_decoder", "lr_milestones", "lr_gamma", "edge_index")
        #embedding
        self.edge_index, self.edge_weight= torch.from_numpy(edge_index).type(torch.LongTensor), torch.from_numpy(edge_weight).float()
        self.embed= EmbedModel(in_channels, embed_out_dim, out_channels, max_diffusion_step, num_nodes)
        # rnn encoder
        enc_in_dim=  self.hparams.embed_out_dim+ 2*self.hparams.y_dim 
        self.rnn_encoder= EncoderRNN(enc_in_dim, self.hparams.r_dim , self.hparams.encoder_num_rnn)

        # rnn_ decoder
        self.rnn_decoder_hosp_inc= DecoderRNN_3(self.hparams.embed_out_dim, self.hparams.z_dim, self.hparams.decoder_hidden_dims, self.hparams.y_dim, self.hparams.decoder_num_rnn)
        self.rnn_decoder_hosp_prev= DecoderRNN_3(self.hparams.embed_out_dim, self.hparams.z_dim, self.hparams.decoder_hidden_dims, self.hparams.y_dim, self.hparams.decoder_num_rnn)
        self.rnn_decoder_latent_inc= DecoderRNN_3(self.hparams.embed_out_dim, self.hparams.z_dim, self.hparams.decoder_hidden_dims, self.hparams.y_dim, self.hparams.decoder_num_rnn)
        self.rnn_decoder_latent_prev= DecoderRNN_3(self.hparams.embed_out_dim, self.hparams.z_dim, self.hparams.decoder_hidden_dims, self.hparams.y_dim, self.hparams.decoder_num_rnn)
        
        #latent encoder
        self.z_encoder= LatentEncoder(self.hparams.r_dim, self.hparams.z_dim)
        #
        self.aggregator= MeanAggregator()

        # criteria
        self.mae= torch.nn.L1Loss(reduction="mean")
        self.mse= torch.nn.MSELoss(reduction="mean")

        # global mu_z, sigma_z
        self.register_buffer("mu_z_global", torch.zeros((self.hparams.seq_len, self.hparams.z_dim), requires_grad=False))
        self.register_buffer("var_z_global", torch.ones((self.hparams.seq_len, self.hparams.z_dim), requires_grad=False))
        
        #stats variables
        self.y_mean= torch.zeros(seq_len, y_dim*NUM_COMP)
        self.y_std= torch.zeros(seq_len, y_dim*NUM_COMP)

        #auxiliary variables, pytorch_ligtning v2.02
        self.validation_step_outputs=[]
        
        self.NUM_COMP= NUM_COMP
        
    def forward(self, x, xt, y0_latent_prev):
        """ 
            @input:
                x:  [B, #nodes, x_dim]
                xt:[B, L, num_Nodes, xt_dim]
                y0: [B, y_dim] 
            @return:
                mu_y, var_y: [L, B, y_dim]
        """
        embed_out= self.get_input_embedding(x, xt)
        zs= self.sample_z(self.mu_z_global.to(y0_latent_prev.device), self.var_z_global.to(y0_latent_prev.device), y0_latent_prev.shape[0])
        mu_post, var_post= self.get_post(y0_latent_prev, embed_out, zs)
        return mu_post, var_post
    
        #
    def update_y_stats(self, y_mean, y_std):
        if isinstance(y_mean, np.ndarray):
            self.y_mean= torch.from_numpy(y_mean).float()
        elif isinstance(y_mean, Tensor):
            self.y_mean= y_mean.float()
        else:
            raise TypeError
        if isinstance(y_std, np.ndarray):
            self.y_std= torch.from_numpy(y_std).float()
        elif isinstance(y_std, Tensor):
            y_std= y_std.float()
        else:
            raise TypeError
        
    def get_latent_tensors(self):
        return self.mu_z_global, self.var_z_global
    
    def get_input_embedding(self, x, xt):
        """
        @input:
            x: [B, #nodes, x_dim]
            xt:[B, L, num_Nodes, xt_dim]
        @return:
            embed_out:[B, L, embed_out_dim]
        """
        embed_out=[] 
        h0= None
        for t in range(self.hparams.seq_len):
            inputs=torch.cat([xt[:, t, ...], x], dim=-1)    
            output, h0= self.embed(inputs, self.edge_index.to(x.device), self.edge_weight.to(x.device), h0)
            embed_out.append(output)
        return torch.stack(embed_out, dim=1)

    def sample_post(self, mu_y, var_y):
        return  mu_y+ torch.randn_like(mu_y)*  var_y.sqrt()

    def sample_z(self, mu_z, var_z, num_samples=1):
        """  
            @input:
                mu_z, var_z: [L, z_dim]
            @return:
                zs: [B, L, z_dim]
        """
        sigma_z= self.get_sigma(var_z)
        normal_samples= torch.distributions.Normal(0, 1).rsample((num_samples, mu_z.shape[0], mu_z.shape[-1])).to(mu_z.device)
        zs= mu_z.unsqueeze(0)+ sigma_z.unsqueeze(0)* normal_samples
        return zs
    
    def get_latent_representation(self, embed_out, y:Tensor, y0:Tensor)->Tensor:
        """
            @ input:
                embed_out: [B, L, embed_out_dim]
                y: [B, L, y_dim]
                y0: [B, y_dim]
            @ output:
            mu_z:[L, z_dim]
            sigma_z:[L, z_dim]

        """
        y= torch.concat([y0.unsqueeze(1), y], dim=1)
        enc_in= torch.cat([embed_out, y[:, 1:, :], y[:, :-1, :]], dim=-1)
        ri = self.rnn_encoder(enc_in)
        # [L, z_dim]
        rs= self.aggregator(ri)
        mu_z, var_z= self.z_encoder(rs)
        return mu_z, var_z

    def get_post(self, y0, embed_out, zs):

        """  
            @return:
                mu_y, vay_y: [B, L, y_dim]
        """
        mu_hosp_inc, var_hosp_inc= self.rnn_decoder_hosp_inc(y0, embed_out, zs)
        mu_hosp_prev, var_hosp_prev= self.rnn_decoder_hosp_prev(y0, embed_out, zs)
        mu_latent_inc, var_latent_inc= self.rnn_decoder_latent_inc(y0, embed_out, zs)
        mu_latent_prev, var_latent_prev= self.rnn_decoder_latent_prev(y0, embed_out, zs)
        mu_post= torch.cat([mu_hosp_inc, mu_hosp_prev, mu_latent_inc, mu_latent_prev], dim=-1)
        var_post= torch.cat([var_hosp_inc, var_hosp_prev, var_latent_inc, var_latent_prev], dim=-1)
        return mu_post, var_post
    
    #
    def loss_fn(self, mu_post:Tensor,
                var_post:Tensor, 
                y_hosp_inc_true:Tensor, 
                y_hosp_prev_true:Tensor,
                y_latent_inc_true:Tensor, 
                y_latent_prev_true:Tensor,
                mu_z_post:Tensor, var_z_post:Tensor, mu_z_prior:Tensor, var_z_prior:Tensor)-> torch.Tensor:

        #[seq_len, z_dim]
        kl= torch.distributions.kl_divergence(dist.Normal(mu_z_post, self.get_sigma(var_z_post)), dist.Normal(mu_z_prior, self.get_sigma(var_z_prior))).sum()
        
        mu_hosp_inc, mu_hosp_prev, mu_latent_inc, mu_latent_prev= torch.chunk(mu_post, self.NUM_COMP, dim=-1)
        var_hosp_inc, var_hosp_prev, var_latent_inc, var_latent_prev= torch.chunk(var_post, self.NUM_COMP, dim=-1)

        nll_hosp_inc= -dist.Normal(mu_hosp_inc, var_hosp_inc.sqrt()).log_prob(y_hosp_inc_true).mean(dim=(1, 2)).mean()
        nll_hosp_prev= -dist.Normal(mu_hosp_prev, var_hosp_prev.sqrt()).log_prob(y_hosp_prev_true).mean(dim=(1, 2)).mean()
        nll_latent_inc= -dist.Normal(mu_latent_inc, var_latent_inc.sqrt()).log_prob(y_latent_inc_true).mean(dim=(1, 2)).mean()
        nll_latent_prev= -dist.Normal(mu_latent_prev, var_latent_prev.sqrt()).log_prob(y_latent_prev_true).mean(dim=(1, 2)).mean()
        nll= nll_hosp_inc+ nll_hosp_prev+ nll_latent_inc+ nll_latent_prev
        self.log_dict({"kl":kl.item(), "nll":nll.item()})
        # -elbo
        neg_elbo= nll + kl
        return neg_elbo

    def configure_optimizers(self):
        optimizer= torch.optim.Adam([{"params":self.rnn_encoder.parameters()}, 
                                    {"params":self.rnn_decoder_hosp_inc.parameters(), "lr": self.hparams.lr_decoder},
                                    {"params":self.rnn_decoder_hosp_prev.parameters(), "lr": self.hparams.lr_decoder}, 
                                    {"params":self.rnn_decoder_latent_inc.parameters(), "lr": self.hparams.lr_decoder},
                                    {"params":self.rnn_decoder_latent_prev.parameters(), "lr": self.hparams.lr_decoder},
                                    {"params":self.z_encoder.parameters()},
                                    {"params":self.embed.parameters()}], lr=self.hparams.lr)

        lr_scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_milestones, gamma= self.hparams.lr_gamma)
        return [optimizer] , [lr_scheduler]


    def on_train_epoch_start(self) -> None:
        self.mu_z_list=[]
        self.var_z_list=[]

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
            batch:(x, xt, y, y0)
            x:[B, #nodes, x_dim]
            xt: [B, L, #nodes, xt_dim]
            y:[B, L, y_dim]
            y0:[B, y_dim] latent prevalence initial value
        """
        # self.to(DEVICE)
        x, xt, y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev, y0_latent_prev= batch
        x_context, xt_context, y_context, y0_context, x_target, xt_target, y_latent_prev_target, y0_target, _, idt= self.context_target_split(x, xt, y_latent_prev, y0_latent_prev, self.hparams.context_percentage)
        y_hosp_inc_target= y_hosp_inc[idt, ...]
        y_hosp_prev_target=y_hosp_prev[idt, ...]
        y_latent_inc_target= y_latent_inc[idt, ...]
        #posterior latent sitributions
        embed_out= self.get_input_embedding(x, xt)

        # mu_z_post, var_z_post= self.get_latent_representation(embed_out, y_latent_prev_normalize, y0)
        mu_z_post, var_z_post= self.get_latent_representation(embed_out, y_latent_prev, y0_latent_prev)
        zs_post= self.sample_z(mu_z_post, var_z_post, y0_target.shape[0])
        #
        embed_out= self.get_input_embedding(x_target, xt_target)
        mu_post, var_post= self.get_post(y0_target, embed_out, zs_post)
        #prior latent distribution
        embed_out= self.get_input_embedding(x_context, xt_context)
        mu_z_prior, var_z_prior= self.get_latent_representation(embed_out, y_context, y0_context)
        #
        self.mu_z_list.append(mu_z_post)
        self.var_z_list.append(var_z_post)
        #
        loss= self.loss_fn(mu_post, var_post, y_hosp_inc_target, y_hosp_prev_target, y_latent_inc_target, y_latent_prev_target, mu_z_post, var_z_post, mu_z_prior, var_z_prior)
        #
        wmape= self.compute_wmape(x, xt, y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev, y0_latent_prev)

        self.log("train_loss", loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_wmape", wmape.item(), on_epoch=True, on_step=False, prog_bar=False)
        return loss
    
    @torch.no_grad()
    def compute_wmape(self, x, xt, y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev, y0_latent_prev):
        
        wmape_fn= WeightedMeanAbsolutePercentageError().to(y_hosp_inc.device)
        embed_out= self.get_input_embedding(x, xt)
        mu_z_post, var_z_post= self.get_latent_representation(embed_out, y_latent_prev, y0_latent_prev)
        zs= self.sample_z(mu_z_post, var_z_post, y0_latent_prev.shape[0])
        mu_post, var_post= self.get_post(y0_latent_prev, embed_out, zs)
        y_post= self.sample_post(mu_post, var_post)
        y= torch.concat([y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev], dim=-1)
        return wmape_fn(y_post, y)
    
    def on_train_epoch_end(self) -> None:
        """  
            @return: [L, y_dim]
        """
        self.mu_z_global= torch.stack(self.mu_z_list, dim=0).mean(0)
        self.var_z_global= torch.stack(self.var_z_list, dim=0).mean(0)
        
        #reset validation output_list
        self.validation_step_outputs=[]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
            batch: (x, y, y0)
            x: [B, #nodes, x_dim]
            xt: [B, L, #nodes, xt_dim]
            y: [B, L, y_dim]
            y0: [B, y_dim]
        """
        # self.to("cpu")
        x, xt, y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev, y0_latent_prev= batch
        y= torch.concat([y_hosp_inc, y_hosp_prev, y_latent_inc, y_latent_prev], dim=-1)
        mu_post, var_post= self(x, xt, y0_latent_prev)
        y_pred= self.sample_post(mu_post, var_post)
        val_mae=  self.mae(y, y_pred)
        #
        wmape_fn= WeightedMeanAbsolutePercentageError().to(y.device)
        wmape= wmape_fn(y_pred, y)
        val_mse= self.mse(y_pred, y)
        self.validation_step_outputs.append({"val_mae":val_mae, "val_wmape":wmape, "val_mse":val_mse})
    
    def on_validation_epoch_end(self) -> None:

        val_mse= torch.stack([x["val_mse"] for x in self.validation_step_outputs], dim=0).mean()
        val_mae= torch.stack([x["val_mae"] for x in self.validation_step_outputs], dim=0).mean()

        val_wmape= torch.stack([x["val_wmape"] for x in self.validation_step_outputs], dim=0).mean()

        self.log("val_mae", val_mae.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_wmape", val_wmape.item(), on_epoch=True, on_step=False, prog_bar=False)
        self.log("val_mse", val_mse.item(), on_epoch=True, on_step=False, prog_bar=False)
        
    @staticmethod
    def context_target_split(x, xt, y, y0,
                        context_perc:float=.2)->Tuple[torch.Tensor, ...]:

        """
            @input:
           	    x: [B, num_nodes, x_dim]
                xt: [B, L, #nodes, xt_dim]
           	    y: [B, L ,y_dim]
                y0:[B, y_dim]
                context_perc: percentage of points to context 
            
            @returns: 
                x_context:[n_context, num_nodes, x_dim] 
                xt_context: [num_context, L, num_nodes, xt_dim]
                y_context: [n_context, seq_len, y_dim] 
                y0_context, x_target, y_target, y0_target
        """
        assert context_perc<1.0 and context_perc>0.0

        B, _, _= y.size()

        assert x.size(0)==B
	
        n_context= int(context_perc*B)

        # context indices
        idx= np.arange(B)
        np.random.shuffle(idx)
        
        idc=idx[:n_context]
        idt= idx[n_context:]

        return x[idc, ...], xt[idc,...], y[idc,... ], y0[idc], x[idt, ...], xt[idt, ...], y[idt, ...], y0[idt], idc, idt

    @torch.no_grad()
    def predict(self, x:Tensor, xt:Tensor, y0_latent_prev:Tensor):
        """
            this method uses the model y_mean and y_std values
            @ input:
                x, xt, y0_latent_prev
            @ return 
                torch tensor : [B, T, ydim]
        """
        assert isinstance(x,  Tensor)
        assert isinstance(xt, Tensor)
        assert isinstance(y0_latent_prev, Tensor)

        *_, y_latent_prev_mean= torch.chunk(self.y_mean, self.NUM_COMP, dim=-1)
        *_, y_latent_prev_std= torch.chunk(self.y_std, self.NUM_COMP, dim=-1)
        
        if x.ndim<3:
            x= x[np.newaxis, ...]

        if y0_latent_prev.ndim<2:
            y0_latent_prev= y0_latent_prev[np.newaxis, ...]    

        y0= self.normalize(y0_latent_prev, y_latent_prev_mean, y_latent_prev_std)

        #[B, #nodes, x_dim]
        mu_post, var_post=self(x, xt, y0)
        y_pred= self.sample_post(mu_post, var_post)
        y_pred= self.unnormalize(y_pred, self.y_mean, self.y_std)
        return y_pred
                
    @torch.no_grad()
    def get_samples(self, x:np.ndarray, xt:np.ndarray, y0_latent_prev:np.ndarray, x_mean, x_std, y_mean, y_std):
        self.update_y_stats(y_mean, y_std)
        
        *_, y_latent_prev_mean= np.split(y_mean, self.NUM_COMP, axis=-1)
        *_, y_latent_prev_std= np.split(y_std, self.NUM_COMP, axis=-1)
        
        if x.ndim<3:
            x= x[np.newaxis, ...]

        if y0_latent_prev.ndim<2:
            y0_latent_prev= y0_latent_prev[np.newaxis, ...]    

        y0= self.normalize(y0_latent_prev, y_latent_prev_mean, y_latent_prev_std)
        
        #[B, #nodes, x_dim]
        x= torch.from_numpy(x).float()
        xt= torch.from_numpy(xt).float()
        y0= torch.from_numpy(y0_latent_prev).float()
        mu_pred, var_pred=self(x, xt, y0)
        samples= self.sample_post(mu_pred, var_pred)
        samples= samples.detach().cpu().numpy()
        samples= self.unnormalize(samples, y_mean, y_std)
        samples[samples<0.0]=0.0
        return np.round(samples)
        
    @staticmethod
    def normalize(data, mean, std):
        eps= 1e-8
        if isinstance(data, torch.Tensor):
            return (data- mean.to(data.device))/(std.to(data.device)+ eps)
        else:
            return (data-mean)/(std+eps)
    @staticmethod
    def unnormalize(data, mean, std):
        eps= 1e-8
        if isinstance(data, torch.Tensor):
            return mean.to(data.device)+ (std.to(data.device)+eps)* data
        else:
            return mean+ (std+eps)* data

    
