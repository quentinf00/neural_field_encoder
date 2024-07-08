import torch
import torch.nn as nn
import einops

class RandomFourierFeatures(nn.Module):
    def __init__(self, n_coords=2, n_features=256, sigma=3.0, coords_scales=1.):
        super().__init__()
        self.B = nn.Parameter(torch.randn(n_coords, n_features//2) * 2 * torch.pi * sigma, requires_grad=False)

    def forward(self, x):
        return torch.cat((torch.cos(x @ self.B), torch.sin(x @ self.B)), dim=-1)



def make_mod(nin, nout=1, nhid=128, ndep=3, sig=10.):
    return nn.Sequential(
        RandomFourierFeatures(n_coords=nin, n_features=nhid, sigma=sig),
        *[m for _ in range(ndep-1) for m in [nn.Linear(nhid, nhid), nn.ReLU()]],
        nn.Linear(nhid, nhid),
        nn.Sigmoid(),
        nn.Linear(nhid, nout)
)


def compute_normstats(data, reduce_pattern='... -> ()', mode='minmax'):
    if mode == 'minmax':
        m = einops.reduce(data, reduce_pattern, 'min')
        M = einops.reduce(data, reduce_pattern, 'max')
        ns = einops.rearrange([m + (M-m)/2, (M-m)/2], '... -> ...')
        return ns 
    else:
       return 1., 0 
    
def compute_normstats_iter(data_iter, reduce_pattern='... -> ()', mode='minmax'):
    if mode == 'minmax':
        m, M = None, None
        for data in data_iter:
            _m = einops.reduce(data, reduce_pattern, 'min')
            _M = einops.reduce(data, reduce_pattern, 'max')
            if m is None:
                m = _m
                M = _M
            else:
                m = einops.reduce([m, _m], 'n ... -> ...', 'min')
                M = einops.reduce([M, _M], 'n ... -> ...', 'min')


        ns = einops.rearrange([m + (M-m)/2, (M-m)/2], '... -> ...')
        return ns 
    else:
       return 1., 0 
