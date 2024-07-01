import torch
import torch.multiprocessing as mp 
from torch.utils.data import DataLoader, SequentialSampler, Subset
import torch.distributions as dist

from activeLearner.active_dataset import ActiveLearningData

import numpy as np


from all_utils.data_utils import pool_collate_fn

class MeanStd:
    def __init__(self, output_dim, acquisition_size, pool_loader_batch_size, acquisition_pool_fraction, num_workers=4, device="cpu"):
        
        self.output_dim= output_dim
        self.acquisition_size= acquisition_size
        self.pool_loader_batch_size= pool_loader_batch_size
        self.acquisition_pool_fraction= acquisition_pool_fraction
        self.num_workers= num_workers
        self.device= device
        self.pin_memory =True

    @torch.no_grad()
    def get_candidate_batch(self, model, active_data:ActiveLearningData, **kwarg):
        
        model.to(self.device)
        pool_size= int(self.acquisition_pool_fraction*len(active_data.pool_dataset))
        pool_indices= torch.randperm(len(active_data.pool_dataset))[:pool_size]
        sampler= SequentialSampler(pool_indices)

        pool_loader= DataLoader(active_data.pool_dataset, shuffle=False, batch_size=self.pool_loader_batch_size, sampler= sampler, num_workers=self.num_workers, collate_fn=pool_collate_fn, pin_memory= self.pin_memory)
        
        scores= torch.zeros(pool_size, dtype=torch.float32, device=self.device)
        
        indices_array=np.zeros(pool_size, dtype=np.int64)

        for i, batch in enumerate(pool_loader):
            
            x, xt, y0, indices= batch
            x, xt, y0= x.to(self.device), xt.to(self.device), y0.to(self.device)

            #[L, B, ydim]
            yp= model.predict(x, xt, y0)
            score= self.acquisition_fn(yp, dim=1)

            ilow=i*self.pool_loader_batch_size
            ihigh= min(ilow+self.pool_loader_batch_size, pool_size)

            scores[ilow:ihigh]= score.cpu()
            
            indices_array[ilow:ihigh]=indices[:]
        #
        best_indices= torch.argsort(scores, dim=0, descending=True)[:self.acquisition_size]

        #back to DEVICE
        # model.to(DEVICE)
        best_indices= best_indices.cpu().numpy()
        return best_indices

    def acquisition_fn(self, y_pred, dim):
        """
            @input: 
                y_pred:[ B, L, y_dim]
        """
        return torch.std(y_pred, dim=dim).mean(dim=-1)
##
class LatentInfoGain:
    acquisition_batch_size:int
    def __init__(self, acquisition_size:int, pool_loader_batch_size , acquisition_pool_fraction:float, num_workers, device="cpu"):
        
        assert acquisition_pool_fraction<= 1.0 and acquisition_pool_fraction>0.0
        assert pool_loader_batch_size%acquisition_size==0

        self.acquisition_size= acquisition_size
        self.pool_loader_batch_size= pool_loader_batch_size
        self.acquisition_pool_fraction= acquisition_pool_fraction
        self.num_workers= num_workers
        self.device= device

    @torch.no_grad()
    def get_candidate_batch(self, model, active_data:ActiveLearningData, **kwarg):
        model.eval().to(self.device)
        pool_size= int(self.acquisition_pool_fraction*len(active_data.pool_dataset))
        pool_indices= torch.randperm(len(active_data.pool_dataset))[:pool_size]

        pool_loader= DataLoader(Subset(active_data.pool_dataset, pool_indices), shuffle=True, batch_size=self.pool_loader_batch_size, num_workers=self.num_workers, collate_fn=pool_collate_fn)

        mu_z_train, var_z_train= model.get_latent_tensors()
        mu_z_train= mu_z_train.to(self.device)
        sigma_z_train= var_z_train.sqrt().to(self.device)

        model=model.to(self.device)
        
        batch_scores=[]
        num_train_data= active_data.train_size 

        for i, batch in enumerate(pool_loader):
            x, xt, y0_latent_prev, indices= batch
           
            x, xt, y0_latent_prev= x.to(self.device), xt.to(self.device), y0_latent_prev.to(self.device)
            
            embed_out= model.get_input_embedding(x, xt)
            y_post= model(x, xt, y0_latent_prev)
            _, _, _, y_latent_prev_post= torch.chunk(y_post, model.NUM_COMP, dim=-1)
            mu_z_q, var_z_q= model.get_latent_representation(embed_out, y_latent_prev_post, y0_latent_prev)
            sigma_z_q= var_z_q.sqrt()
            score= self.acquisition_fn(mu_z_train, sigma_z_train, mu_z_q, sigma_z_q, x.size(1), num_train_data)
            batch_scores.append((score, indices))
        #
        _, candidate_indices= max(batch_scores, key=lambda x: x[0])
        return candidate_indices
    
    def acquisition_fn(self, mu_z_train, sigma_z_train, mu_z_q, sigma_z_q, query_size, num_train_data)->float:
        denum= query_size+ num_train_data
        mu_z_post = (mu_z_q * query_size + mu_z_train * num_train_data)/denum 
        sigma_z_post = torch.sqrt((sigma_z_q**2.0 * query_size+ sigma_z_train**2.0 * num_train_data)/denum) 
        mu_z_post = mu_z_post.to(self.device)
        sigma_z_post= sigma_z_post.to(self.device)
        normal_q= dist.Normal(mu_z_post, sigma_z_post)
        normal= dist.Normal(mu_z_train,  sigma_z_train)
        score= dist.kl_divergence(normal_q, normal).sum()
        return score.item()
#####

class LatentInfoGainStream:
    acquisition_batch_size:int
    def __init__(self, acquisition_size:int, pool_loader_batch_size , acquisition_pool_fraction:float, num_workers, device="cpu"):
        
        assert acquisition_pool_fraction<= 1.0 and acquisition_pool_fraction>0.0
        assert pool_loader_batch_size%acquisition_size==0

        self.acquisition_size= acquisition_size
        self.pool_loader_batch_size= pool_loader_batch_size
        self.acquisition_pool_fraction= acquisition_pool_fraction
        self.num_workers= num_workers
        self.device= device

    @torch.no_grad()
    def get_candidate_batch(self, model, active_data:ActiveLearningData, **kwarg):
        model.eval().to(self.device)
        pool_size= int(self.acquisition_pool_fraction*len(active_data.pool_dataset))
        pool_indices= torch.randperm(len(active_data.pool_dataset))[:pool_size]

        pool_loader= DataLoader(Subset(active_data.pool_dataset, pool_indices), shuffle=True, batch_size=self.pool_loader_batch_size, num_workers=self.num_workers, collate_fn=pool_collate_fn)

        mu_z_train, var_z_train= model.get_latent_tensors()
        mu_z_train= mu_z_train.to(self.device)
        sigma_z_train= var_z_train.sqrt().to(self.device)

        model=model.to(self.device)
        
        batch_scores=[]
        num_train_data= active_data.train_size 

        for i, batch in enumerate(pool_loader):
            x, xt, y0_latent_prev, indices= batch
           
            x, xt, y0_latent_prev= x.to(self.device), xt.to(self.device), y0_latent_prev.to(self.device)
            x_chunk= x.chunk(self.acquisition_size, dim=0)
            xt_chunk = xt.chunk(self.acquisition_size, dim=0)
            y0_chunk= y0_latent_prev.chunk(self.acquisition_size, dim=0)
            idx_chunk= np.split(indices, self.acquisition_size, axis=0)
            scores= self.process_batch_cuda(model, x_chunk, xt_chunk, y0_chunk, mu_z_train, sigma_z_train, num_train_data)
            
            batch_scores+=list(zip(scores, idx_chunk))
        #
        _, candidate_indices= max(batch_scores, key=lambda x: x[0])
        return candidate_indices
    
    def process_batch_cuda(self, model, x_chunk, xt_chunk, y0_chunk, mu_z_train, sigma_z_train, num_train_data):
        streams= [torch.cuda.Stream() for _ in range(len(x_chunk))]
        scores=[]
        for stream, x, xt, y0 in zip(streams, x_chunk, xt_chunk, y0_chunk):
            with torch.cuda.stream(stream):
                score=self.process_chunk(self, model, x, xt, y0, mu_z_train, sigma_z_train, num_train_data)
                scores.append(score.detach().cpu().item())
        return scores

    def process_chunk(self, model, x, xt, y0, mu_z_train, sigma_z_train, num_train_data):
        embed_out= model.get_input_embedding(x, xt)
        y_post= model(x, xt, y0)
        _, _, _, y_latent_prev_post= torch.chunk(y_post, model.NUM_COMP, dim=-1)
        mu_z_q, var_z_q= model.get_latent_representation(embed_out, y_latent_prev_post, y0)
        sigma_z_q= var_z_q.sqrt()
        score= self.acquisition_fn(mu_z_train, sigma_z_train, mu_z_q, sigma_z_q, x.size(1), num_train_data) 
        return score
    
    def acquisition_fn(self, mu_z_train, sigma_z_train, mu_z_q, sigma_z_q, query_size, num_train_data)->float:
        denum= query_size+ num_train_data
        mu_z_post = (mu_z_q * query_size + mu_z_train * num_train_data)/denum 
        sigma_z_post = torch.sqrt((sigma_z_q**2.0 * query_size+ sigma_z_train**2.0 * num_train_data)/denum) 
        mu_z_post = mu_z_post.to(self.device)
        sigma_z_post= sigma_z_post.to(self.device)
        normal_q= dist.Normal(mu_z_post, sigma_z_post)
        normal= dist.Normal(mu_z_train,  sigma_z_train)
        score= dist.kl_divergence(normal_q, normal).sum()
        return score.item()

    def process_batch_cpu(self, model, x, xt, y0, mu_z_train, sigma_z_train, num_train_data):
        embed_out= model.get_input_embedding(x, xt)
        y_post= model(x, xt, y0)
        _, _, _, y_latent_prev_post= torch.chunk(y_post, model.NUM_COMP, dim=-1)
        mu_z_q, var_z_q= model.get_latent_representation(embed_out, y_latent_prev_post, y0)
        sigma_z_q= var_z_q.sqrt()
        score= self.acquisition_fn(mu_z_train, sigma_z_train, mu_z_q, sigma_z_q, x.size(1), num_train_data) 
        return score


    
