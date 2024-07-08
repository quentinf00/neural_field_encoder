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


### End Clean
def fit_diff(fn, params, dl, max_epochs=1, max_steps=10, loss_fn=F.mse_loss, opt=torchopt.adam(lr=2e-4)):
    optimizer = torchopt.FuncOptimizer(opt)

    for _ in range(max_epochs):
        for i, (x, y) in enumerate(dl):
            yhat = fn(params, x)
            loss = loss_fn(yhat, y)
            optimizer.step(loss, params)

            if i == max_steps:
                break

    return params

class NeuralFieldEncoder(nn.Module):
    def __init__(self, base_nf, nndecoder):
        super().__init__(self)
        self.base_nf = base_nf
        self.nndecoder = nndecoder
        self.code_init = nn.Parameter(torch.zeros(nn.decoder.dcode))
        

    def init_code(self, bs=None):
        if bs is None:
            return self.code_init.clone()
        else:
            return torch.stack([self.code_init.clone() for _ in range(bs)], dim=0)

    def nf(self):
        state_dict = self.nndecoder(code)
        return partial(torch.func.functional_call, self.base_nf, state_dict)

    def batch_nf(self, code):
        state_dict = self.nndecoder(code)
        return partial(torch.vmap(partial(torch.func.functional_call, self.base_nf)), state_dict)

    def forward(self, batch):
        coords, values = batch
        pass


class LitNeuralFieldEncoder(L.LightningModule):
    def __init__(self, dcode, nsamples):
        self.register_buffer('samples_codes', torch.zeros(nsamples, dcode))

        pass

### End WIP
def encode(
        nf_encoder,
        coords,
        values,
        dl_kws=dict(batch_size=512),
        loss_fn=F.mse_loss,
    ):
    code = nf_encoder.init_code()
    dl = tdat.DataLoader(tdat.TensorDataset(coords, values), **dl_kws)

    opt = torchopt.adam(lr=2e-4)
    state = opt.init((code,))
    for epoch in range(max_epochs):
    for batch in dl:

    nf = nf_encoder.load_nf(code=code)
    for step in range(self.inner_steps):
        loss = loss_fn(preds, values)
        grads = torch.autograd.grad(loss, (latents,))
        updates, state = opt.update(grads, state, inplace=False)
        (latents,) = torchopt.apply_updates((latents,), updates)
    return 

def deco













class NF(L.LightningModule):
    def __init__(self, mod, loss_fn=F.mse_loss, lr=5e-4):
        self.mod = mod
        self.loss_fn = loss_fn
        self.lr = lr
        self.state = self.state_init()
   
    def state_init(self):
        return None
    
    def eval_nf(self, coords, state=None):
        state= state or self.state
        pass

    def training_step(self, batch, batch_idx):
        coords, vals = batch
        vals_hat = self.eval_nf(coords)
        loss = self.loss_fn(vals, vals_hat)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        coords, vals = batch
        vals_hat = self.eval_nf(coords)
        self.log('val_mse', F.mse_loss(vals, vals_hat), on_step=False, on_epoch=True, prog_bar=True)
        return  self.loss_fn(vals, vals_hat)
    
    def predict_step(self, batch, batch_idx):
        return self.mod(batch[0]).detach().cpu().numpy()
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
        return dict(optimizer=opt, lr_scheduler=sched)
   




def eval_nf(state, coords):
    pass
    
def fit_nf_state(
        state_init,
        nf_fn,
        dl,
        opt,
        max_epochs
    ):
    pass

class MetaNF(L.LightningModule):
    def __init__(self, mod, outer_batch_prop=0.2, inner_batch_prop=0.1, inner_steps=3):
        super().__init__()
        self.mod = mod
        self.inner_batch_prop = inner_batch_prop
        self.outer_batch_prop = outer_batch_prop
        self.inner_steps = inner_steps
        self.p_shape = {k: p.shape for k,p in self.mod.named_parameters()}
        self.p_numel = {k: p.numel() for k,p in self.mod.named_parameters()}
        self.n_lat = sum(self.p_numel.values())
        self.latent_init = nn.Parameter(torch.concat(tuple([p.ravel() for p in self.mod.parameters()])))

    def __repr__(self):
        return f"""
        {self.mod=}
        
        {self.p_shape=}
        
        {self.n_lat=}
        
        {self.latent_init=}"""
    
    def init_latent(self, bs=None):
        if bs is None:
            return self.latent_init.clone()
        else:
            return torch.stack([self.latent_init.clone() for _ in range(bs)], dim=0)

    def latent_to_params(self, latent):
        state_dict = dict()
        p_st = 0
        for k, numel, shape in zip(self.p_shape.keys(), self.p_numel.values(), self.p_shape.values()):
            p_flat = latent[..., p_st:p_st + numel]
            new_shape = p_flat.shape[:-1] + shape
            state_dict[k] = p_flat.reshape(new_shape)
            p_st += numel
        return state_dict

    def sub_samp(self, batch, prop=None):
        coords, quantities, idxes = batch
        samp = torch.rand(coords.shape[-2:]) < self.inner_batch_prop
        coords, quantities, _ = batch
        return coords[:, :, *samp.nonzero(as_tuple=True)], quantities[:, :, *samp.nonzero(as_tuple=True)]
        
    def training_step(self, batch, inner_steps=None):
        coords, quantities, idxes = batch
        bs = coords.shape[0]
        latents = self.init_latent(bs=bs)
        
        opt = torchopt.adam(lr=2e-4)
        state = opt.init((latents,))
        inner_loss = F.mse_loss
        outer_loss = F.mse_loss

        inner_steps = inner_steps if inner_steps is None else self.inner_steps
        for step in range(self.inner_steps):
            inner_coords, inner_quantities = map(
                partial(einops.rearrange, pattern='b d n -> b n d'),
                self.sub_samp(batch, prop=self.inner_batch_prop)
            )
            params = self.latent_to_params(latents)
            out = torch.func.vmap(partial(torch.func.functional_call, self.mod))(params, inner_coords)
            loss = inner_loss(out, inner_quantities)
            grads = torch.autograd.grad(loss, (latents,))
            updates, state = opt.update(grads, state, inplace=False)
            (latents,) = torchopt.apply_updates((latents,), updates)

        sub_coords, sub_quantities = map(
            partial(einops.rearrange, pattern='b d n -> b n d'),
            self.sub_samp(batch, prop=self.inner_batch_prop)
        )
        params = self.latent_to_params(latents)
        out = torch.func.vmap(partial(torch.func.functional_call, self.mod))(params, sub_coords)
        return outer_loss(out, sub_quantities)

    def forward(self, coords, latent=None):
        if latent is None:
            latent = self.init_latent()
        params = self.latent_to_params(latent)
        return torch.func.functional_call(self.mod, params, coords)


def reshape_n_back(src_shape, tgt_shape, fwd_kws=dict(), bwd_kws=dict())
    fwd = partial(einops.rearrange, pattern=src_shape + ' -> ' + tgt_shape, **fwd_kws)
    bwd = partial(einops.rearrange, pattern=tgt_shape + ' -> ' + src_shape, **fwd_kws)
    return fwd, bwd


def pad_stack(ts, dim=0):
    """
    Stack tensors with different shapes padding with nan the smaller ones
    """
    tgt_size = torch.stack([torch.tensor(t.shape) for t in ts]).max(0).values.long()
    ps = lambda t: torch.stack(
            [torch.zeros_like(tgt_size), (tgt_size - torch.tensor(t.size())).maximum(torch.zeros_like(tgt_size))]
            , dim=-1).flatten().__reversed__()
    return torch.stack([F.pad(t, [x.item() for x in ps(t)], value=np.nan) for t in ts], dim=dim)

tdat.TensorDataset(pad_stack([torch.zeros(10, 10),torch.zeros(100, 10)]))

def crop_stack(ts, dim=0):
    """
    Stack tensors with different shapes padding with nan the smaller ones
    """
    tgt_size = torch.stack([torch.tensor(t.shape) for t in ts]).min(0).values.long()
    sl = tuple([slice(None, s) for s in tgt_size])
    return torch.stack([t[sl] for t in ts], dim=dim)

len(tdat.TensorDataset(crop_stack([torch.zeros(10, 10),torch.zeros(100, 10)])))

