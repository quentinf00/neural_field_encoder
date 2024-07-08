import torch
import torch.func
import torchopt
from functools import partial
import einops
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch.utils.data as tdat
import numpy as np


class NNBasisDecoder(nn.Module):
    def __init__(self, named_params, dim_code=512, nbasis=128):
        super().__init__()
        self.p_shape = {k: p.shape for k,p in named_params}
        self.p_numel = {k: p.numel() for k,p in named_params}
        # n_params = sum(self.p_numel.values())
        # self.basis = nn.Parameter(torch.randn(dbasis, n_params))
        flat_params = torch.concat(tuple([p.ravel() for _, p in named_params]))
        self.basis = nn.Parameter(torch.stack([flat_params for _ in range(nbasis)], dim=0))
        self.decoder = nn.Sequential(
            nn.Linear(dim_code, nbasis),
            nn.Softmax(dim=-1)
        )

    def forward(self, code):
        # flat_params = self.decoder(code) @ self.basis / self.basis.sum(0, keepdim=True)
        # print(f'{code.shape=}')
        # print(f'{self.decoder(code).shape=}')
        # print(f'{self.basis.shape=}')
        # print()
        flat_params =(self.decoder(code)[..., None] * self.basis).sum(-2)
        # print(f'{flat_params.shape=}')
        state_dict = dict()
        p_st = 0
        for k, numel, shape in zip(self.p_shape.keys(), self.p_numel.values(), self.p_shape.values()):
            p_flat = flat_params[..., p_st:p_st + numel]
            new_shape = p_flat.shape[:-1] + shape
            state_dict[k] = p_flat.reshape(new_shape)
            p_st += numel
        return state_dict


if __name__ == '__main__':
    import lorenz63
    import nf
    import hydra
    import einops
    import utils
    import time
    import importlib
    importlib.reload(utils)

    data = hydra.utils.call(lorenz63.traj_cfg)


    coords, values = torch.as_tensor(data.time.values), torch.as_tensor(data.values)

    coords = coords.float()
    values = values.float()

    coords = einops.rearrange(coords, 'time -> time ()')
    values = einops.rearrange(values, 'comp time -> time comp')

    # chunk data
    nchunk =  20
    c_chunks = einops.rearrange(coords, '(nchunk tchunk) ... -> nchunk tchunk ...', nchunk=nchunk)
    v_chunks = einops.rearrange(values, '(nchunk tchunk) ... -> nchunk tchunk ...', nchunk=nchunk)
    c_chunks_start = einops.reduce(c_chunks, 'nchunk tchunk () -> nchunk () ()', 'min')
    c_chunks_inputs = c_chunks - c_chunks_start

    cns = utils.compute_normstats(c_chunks_inputs, 'nchunk tchunk () -> ()')
    vns = utils.compute_normstats(v_chunks, 'nchunk tchunk comp-> comp')

    # instatiate models
    mod = utils.make_mod(1, 3, nhid=256)
    mod_nf = nf.NeuralField(mod, cns, vns)

    dcode = 64
    dbasis = 2
    mod_nfb = NNBasisDecoder([*mod_nf.named_parameters()], dim_code=dcode, nbasis=dbasis)

    codes = torch.randn((nchunk, dcode), requires_grad=True)

    device = 'cuda'
    codes = codes.to(device).detach().requires_grad_()
    mod_nfb = mod_nfb.to(device)
    c_chunks_inputs = c_chunks_inputs.to(device)
    v_chunks = v_chunks.to(device)
    mod_nf = mod_nf.to(device)
    fn_nf = partial(torch.func.functional_call, mod_nf)
    vmapped_nf = torch.vmap(fn_nf)


    base_lr = 5e-6
    opt = torch.optim.Adam(
        [
            dict(params=(codes,), lr=2*base_lr),
            dict(params=mod_nfb.parameters(), lr=base_lr),
        ]
        , lr=base_lr)

    t0 = time.time()
    for step in range(2000):
        state_dicts = mod_nfb(codes)
        out = vmapped_nf(state_dicts, c_chunks_inputs)
        l = nn.functional.mse_loss(out, v_chunks)
        opt.zero_grad()
        l.backward()
        opt.step()
        print(f"t: {time.time() -t0: .0f} Epoch: {step}, loss {l.detach().item()}", end='\r')


    flat_out = einops.rearrange(out, 'nchunk tchunk ... -> (nchunk tchunk) ...', nchunk=nchunk) 
    ds = data.to_dataset(name='true').assign(nf=(('time', 'component'), flat_out.detach().cpu()))
    ds.to_dataframe().assign(err=lambda df: df.true - df.nf).reset_index().groupby('component')[['true', 'err']].describe()

