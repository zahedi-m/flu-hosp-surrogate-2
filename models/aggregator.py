import torch

class MeanAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r:torch.Tensor):
        # r:[B, L, embed_out_dim]
        return r.mean(dim=0)