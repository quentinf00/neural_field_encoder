import torch
import torch.func
import torch.nn as nn
import torch.utils.data as tdat

class NeuralField(nn.Module):
    def __init__(self, mod, coords_normstats=(1., 0), values_normstats=(1., 0), **kwargs):
        super().__init__()
        self.register_buffer('coords_normstats', torch.as_tensor(coords_normstats))
        self.register_buffer('values_normstats', torch.as_tensor(values_normstats))
        self.mod = mod

    def process_coords(self, coords):
        offset, scale = self.coords_normstats
        return (coords - offset) / scale

    def process_values(self, values):
        offset, scale = self.values_normstats
        return values * scale + offset

    def forward(self, coords, process_coords=True, process_values=True):
        if process_coords:
            coords = self.process_coords(coords)

        values = self.mod(coords)

        if process_values:
            values = self.process_values(values)
        return values


if __name__ == '__main__':
    import lorenz63
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

    cns = utils.compute_normstats(coords)
    vns = utils.compute_normstats(values, 'time comp-> comp')
    mod = utils.make_mod(1, 3, nhid=256)
    nf = NeuralField(mod, cns, vns)

    opt = torch.optim.Adam(nf.parameters(), lr=1e-5)

    t0 = time.time()
    for step in range(500):
        out = nf(coords)
        l = nn.functional.mse_loss(out, values)
        opt.zero_grad()
        l.backward()
        opt.step()
        print(f"t: {time.time() -t0: .0f} Epoch: {step}, loss {l.detach().item()}", end='\r')


    ds = data.to_dataset(name='true').assign(nf=(('time', 'component'), out.detach().cpu()))
    ds.to_dataframe().assign(err=lambda df: df.true - df.nf).reset_index().groupby('component')[['true', 'err']].describe()

