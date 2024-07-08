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


def coords_fn(da):
    c = (da.time - da.time.min())
    c = c.values.astype(np.float32)
    c = einops.rearrange(c, 'time -> time ()')
    return c

def vals_fn(da):
    v = da.values.astype(np.float32)
    v = einops.rearrange(v, 'comp time -> time comp')
    return v

class PatcherDataset:
    def __init__(self, patcher, coords_fn, vals_fn):
        self.patcher = patcher
        self.coords_fn = coords_fn
        self.vals_fn = vals_fn

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, idx):
        return idx, self.coords_fn(self.patcher[idx]), self.vals_fn(self.patcher[idx])


class LitNeuralFieldEncoder(L.LightningModule):
    def __init__(self, mod_nf, nn_decoder, dim_code, n_train_samples, inner_lr=1e-3):
        super().__init__()
        self.mod_nf = mod_nf
        self.nn_decoder = nn_decoder
        self.dim_code = dim_code
        self.register_buffer('train_samples_codes', torch.randn(n_train_samples, dim_code), persistent=False)
        self.code_init = nn.Parameter(torch.randn(dim_code))
        self.inner_lr = inner_lr
        self.n_test_iter = 50


        
        self.fn_nf = partial(torch.func.functional_call, mod_nf) # pyright: ignore
        self.vmapped_nf = torch.vmap(self.fn_nf)
        
    def encode(self, coords, vals, codes_init=None, niter=10, lr=1e-3, training=False):
        if codes_init is None:
            codes_init = torch.stack([self.code_init for _ in range(coords.shape[0])], dim=0)

        with torch.set_grad_enabled(True):
            codes = codes_init.requires_grad_()
            optimizer = torchopt.adam(lr=lr)
            opt_state = optimizer.init((codes,))                       # init optimizer
            for _ in range(niter):
                state_dicts = self.nn_decoder(codes)
                out = self.vmapped_nf(state_dicts, coords)
                msk = vals.isfinite()
                l = nn.functional.mse_loss(out.nan_to_num()[msk], vals.nan_to_num()[msk])
                grads = torch.autograd.grad(l, (codes,), create_graph=training)                # compute gradients
                grads = tuple([torch.clamp(g, -0.03, 0.03) for g in grads])
                updates, opt_state = optimizer.update(grads, opt_state)  # get updates
                codes, = torchopt.apply_updates((codes,), updates, inplace=(not training))

            state_dicts = self.nn_decoder(codes)
            out = self.vmapped_nf(state_dicts, coords)
            msk = vals.isfinite()
            l = nn.functional.mse_loss(out.nan_to_num()[msk], vals.nan_to_num()[msk])
        return l, codes, out

    def training_step(self, batch, batch_idx):

        idxes, coords, vals = batch
        buf_codes = self.train_samples_codes[idxes]
        init_codes = torch.stack([self.code_init for _ in range(len(idxes))], dim=0)


        n_epochs_warmup = 200
        n_epochs_warmup_wait = 50

        ep = max(0., self.current_epoch - n_epochs_warmup_wait)
        prop_warmup = min(ep/n_epochs_warmup, 1.)

        n_iter_start, n_iter_end = 0, 10
        n_iter = n_iter_start + (prop_warmup * (n_iter_end - n_iter_start))//1
        codes = prop_warmup * init_codes + buf_codes * (1 - prop_warmup)

        l, codes, _ = self.encode(coords, vals, codes_init=codes, niter=int(n_iter), lr=self.inner_lr, training=True)

        self.log('train_rmse', l**2, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train_iter', n_iter, on_epoch=True, on_step=False, prog_bar=True)
        return l

    def validation_step(self, batch, batch_idx):
        _, coords, vals = batch

        l, *_ = self.encode(coords, vals, niter=10, lr=self.inner_lr)
        self.log('val_rmse', l**2, on_epoch=True, on_step=False, prog_bar=True)
        return l


    def test_step(self, batch, batch_idx, dataloader_idx=0):

        _, coords, vals = batch
        l, *_ = self.encode(coords, vals, niter=self.n_test_iter, lr=self.inner_lr)
        self.log('test_rmse', l**2, on_epoch=True, on_step=False, prog_bar=True)
        return l


    def predict_step(self, batch, batch_idx):

        _, coords, vals = batch
        l, codes, out, = self.encode(coords, vals, niter=self.n_test_iter, lr=self.inner_lr)
        return codes.detach().cpu(), out.detach().cpu()

    def configure_optimizers(self): # pyright: ignore
        opt = torch.optim.Adam(
            [
                dict(params=self.parameters(), lr=1e-3),
                dict(params=self.buffers(), lr=1e-3),
            ], lr=1e-4)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[ 10, 50, 100, 250, 1000, 2500], gamma=0.7)
        return [opt], [sched]


class LitNeuralFieldDiag(LitNeuralFieldEncoder):
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        (_, coords, vals), (_, test_coords, test_vals) = batch
        _, codes, _ = self.encode(coords, vals, niter=self.n_test_iter, lr=self.inner_lr)
        state_dicts = self.nn_decoder(codes)
        out = self.vmapped_nf(state_dicts, test_coords)
        msk = test_vals.isfinite()
        l = nn.functional.mse_loss(out.nan_to_num()[msk], test_vals.nan_to_num()[msk])
        self.log('test_rmse', l**2, on_epoch=True, on_step=False, prog_bar=True)
        return l

    
if __name__ == '__main__':
    import lorenz63
    import neural_field
    import nn_basis_decoder
    import hydra
    import lightning.pytorch.callbacks as Lcb
    import toolz
    import einops
    import utils
    import time
    import importlib
    importlib.reload(utils)
    importlib.reload(nn_basis_decoder)
    import xrpatcher

    data = hydra.utils.call(lorenz63.traj_cfg)
    data = data.astype(np.float32).assign_coords(time=data.time.astype(np.float32))


    splits = dict(train=slice(0, 100), val=slice(100, 150), test=slice(150, 200))

    patcher_fn = lambda da: xrpatcher.XRDAPatcher(
        da, patches=dict(time=100), strides=dict(time=50))
    ds_fn = lambda pa: PatcherDataset(pa, coords_fn=coords_fn, vals_fn=vals_fn)
    dses = {k: ds_fn(patcher_fn(data.sel(time=sl))) for k, sl in splits.items()}

    dl_fn = lambda ds, k: tdat.DataLoader(ds, shuffle=(k=='train'), batch_size=16, num_workers=6)
    dls = { k: dl_fn(ds, k) for k, ds in dses.items()}

    print(f'{[(k, len(ds)) for k,ds in dses.items()]}')
    print(f'{[(k, len(dl)) for k,dl in dls.items()]}')
    batch = next(iter(dls['train']))
    idxes, coords, vals = batch
    trainer = L.Trainer(
        accelerator='gpu',
        logger=True,
        callbacks=[
            Lcb.ModelCheckpoint(monitor='val_rmse', save_top_k=3),
            Lcb.GradientAccumulationScheduler(scheduling={1000: 2, 2000: 4, 5000: 8})
        ],
        devices=1,
        inference_mode=False,
        check_val_every_n_epoch=5,
        max_epochs=5000,
        gradient_clip_val=0.5,
    )

    cns = utils.compute_normstats_iter(map(lambda b: b[1], dls['train']))
    vns = utils.compute_normstats_iter(map(lambda b: b[2], dls['train']), '... comp-> comp')


    mod = utils.make_mod(1, 3, nhid=256, ndep=3)
    mod_nf = neural_field.NeuralField(mod, cns, vns)
    dim_code = 1024
    nbasis = 128
    mod_nfb = nn_basis_decoder.NNBasisDecoder(
        [*mod_nf.named_parameters()],
        dim_code=dim_code,
        nbasis=nbasis
    )
    lit_nfe =  LitNeuralFieldEncoder(
        mod_nf=mod_nf,
        nn_decoder=mod_nfb,
        dim_code=dim_code,
        n_train_samples=len(dses['train']),
    )
    # ckpt = lit_nfe.state_dict()
    # lit_nfe.load_state_dict(ckpt)

    # trainer.fit(lit_nfe, train_dataloaders=dls['train'], val_dataloaders=dls['val'])
    # trainer.test(lit_nfe, dataloaders=dls['test'], ckpt_path='best')

    # trainer.test(lit_nfe, dataloaders=dls['val'], ckpt_path='best_ckpt_sofar_256hid_128basis_1024code.pt')

    lit_nfe =  LitNeuralFieldDiag.load_from_checkpoint('best_ckpt_sofar_256hid_128basis_1024code.pt',
        mod_nf=mod_nf,
        nn_decoder=mod_nfb,
        dim_code=dim_code,
        n_train_samples=len(dses['train']),
    )

    transforms = dict(
        only_first=lorenz63.only_first_obs,
        subsamp=lorenz63.subsample,
        noisy=lorenz63.add_noise,
        all=toolz.compose_left(
            lorenz63.only_first_obs,
            lorenz63.subsample,
            lorenz63.add_noise,
        )
    )

    split = 'val'
    tr = transforms['all']
    diag_ds = tdat.StackDataset(
            ds_fn(patcher_fn(tr(data).sel(time=splits[split]))),
            dses['val']
    )
    diag_dl = dl_fn(diag_ds, 'val')
    trainer.test(lit_nfe, dataloaders=diag_dl, ckpt_path='best_ckpt_sofar_256hid_128basis_1024code.pt')

    # preds = trainer.predict(lit_nfe, dataloaders=diag_dl, ckpt_path='best_ckpt_sofar_256hid_128basis_1024code.pt')
    #
    #
    # codes, outs = [torch.cat(ts) for ts in zip(*preds)]
    # outs.shape
    # samp = dses[split].patcher[0]
    # samp = samp.to_dataset(name='gt').assign(pred = (('time', 'component'), outs[0].detach().numpy()))
    # (samp.gt - samp.pred).pipe(lambda err: err**2).mean('time').pipe(np.sqrt)
    #


