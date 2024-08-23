import abc
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch import nn
import torchvision.transforms as T

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.ops.losses.histogram_loss import interlevel_loss
from plenoxels.raymarching.ray_samplers import RaySamples


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)

    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()

    # h_tv = torch.pow((t[..., 1:, :] - t[..., :h-1, :]).abs(), 3).sum()
    # w_tv = torch.pow((t[..., :, 1:] - t[..., :, :w-1]).abs(), 3).sum()

    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg

def compute_plane_smoothness(t):
 
    batch_size, c, h, w = t.shape
    
    # # # Convolve with a second derivative filter, in the time dimension which is dimension 2    
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    
    # first_difference = t[..., 1:] - t[..., :w-1]  # [batch, c, h-1, w]
    # second_difference = first_difference[..., 1:] - first_difference[..., :w-2]  # [batch, c, h-2, w]
    
    return torch.square(second_difference).mean() # torch.square(second_difference).mean() + 

def compute_plane_smoothness_wavelet(t):

    if len(t.shape) > 4:
        batch_size, c, b, h, w = t.shape
        horiz = t[..., 0, :, :]
        vert = t[..., 1, :, :]
        diag = t[..., 2, :, :]
        
        # batch_size, c, h, w = horiz.shape

        # Along width so is time axis
        # first_difference = horiz[..., 1:] - horiz[..., :w-1]  # [batch, c, h-1, w]
        # second_difference = first_difference[..., 1:] - first_difference[..., :w-2]  # [batch, c, h-2, w]
        # total = torch.pow(second_difference.abs(), 2).mean() #torch.square(second_difference).mean()

        # Both axis
        first_difference = diag[..., 1:] - diag[..., :w-1]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:] - first_difference[..., :w-2]  # [batch, c, h-2, w]
        total =  torch.pow(second_difference.abs(), 2).mean() #torch.square(second_difference).mean()


        first_difference = diag[..., 1:, :] - diag[..., :h-1, :]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]
        total +=  torch.pow(second_difference.abs(), 2).mean() #torch.square(second_difference).mean()

        # Along spatial axis
        # first_difference = vert[..., 1:, :] - vert[..., :h-1, :]  # [batch, c, h-1, w]
        # second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]
        # total = torch.pow(second_difference.abs(), 2).mean() #torch.square(second_difference).mean()


       # first_difference = horiz[..., 1:, :] - horiz[..., :h-1, :]  # [batch, c, h-1, w]
        # second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]
        # total +=  torch.pow(second_difference.abs(), 3).mean() #torch.square(second_difference).mean()

        # Along height
        # first_difference = horiz[..., 1:, :] - horiz[..., :h-1, :]  # [batch, c, h-1, w]
        # second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]
        # total +=  torch.pow(second_difference.abs(), 3).mean() # torch.square(second_difference).mean()

        return total
    else:
        batch_size, c, h, w = t.shape

        # Convolve with a second derivative filter, in the time dimension which is dimension 2    
        first_difference = t[..., 1:] - t[..., :w-1]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:] - first_difference[..., :w-2]  # [batch, c, h-2, w]
        total = torch.pow(second_difference.abs(), 2).mean()
        
        # Along spatial axis
        first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
        total+= torch.pow(second_difference.abs(), 2).mean()
        # Lp Loss
        return total

        # return torch.square(second_difference).mean()

class Regularizer():
    def __init__(self, reg_type, initialization):
        self.reg_type = reg_type
        self.initialization = initialization
        self.weight = float(self.initialization)
        self.last_reg = None

    def step(self, global_step):
        pass

    def report(self, d):
        if self.last_reg is not None:
            d[self.reg_type].update(self.last_reg.item())

    def regularize(self, *args, **kwargs) -> torch.Tensor:
        out = self._regularize(*args, **kwargs) * self.weight
        self.last_reg = out.detach()
        return out

    @abc.abstractmethod
    def _regularize(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def _regularize_with_deformation(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self):
        return f"Regularizer({self.reg_type}, weight={self.weight})"


class PlaneTV(Regularizer):
    def __init__(self, initial_value, what: str = 'field', info=None):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'planeTV-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what
        self.info = info

    def step(self, global_step):
        pass

    def _regularize(self, model: LowrankModel, **kwargs):
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            if self.info['field']: multi_res_grids = model.field.grids()
            else: multi_res_grids = model.field.grids

        elif self.what == 'proposal_network':
            if self.info['psn']: 
                multi_res_grids = [p.grids() for p in model.proposal_networks]
                multi_res_grids = multi_res_grids[0] + multi_res_grids[1] 
            else: multi_res_grids = [p.grids for p in model.proposal_networks]

        else:
            raise NotImplementedError(self.what)
        total = 0

        # Note: input to compute_plane_tv should be of shape [batch_size, c, h, w]
        for grids in multi_res_grids:
            if len(grids) == 3:
                spatial_grids = [0, 1, 2]
            else:
                spatial_grids = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal
            for grid_id in spatial_grids:
                total += compute_plane_tv(grids[grid_id])
            for grid in grids:
                total += compute_plane_tv(grid)
       
        return  torch.as_tensor(total)
    

class TimeSmoothness(Regularizer):
    def __init__(self, initial_value, what: str = 'field', info=None):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'time-smooth-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what
        self.info = info

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            if self.info['field']:
                # Smooth highest freq wavelet coefficients and tend to 0
                multi_res_grids = model.field.grids(time_only=True) #, regularise_wavelet_coeff=True ,notflat=True) #(regularise_wavelet_coeff=True)
            else: multi_res_grids = model.field.grids

        elif self.what == 'proposal_network':
            if self.info['psn']: 
                multi_res_grids = [p.grids(time_only=True) for p in model.proposal_networks] #, regularise_wavelet_coeff=True, notflat=True)
                multi_res_grids = multi_res_grids[0] + multi_res_grids[1]

            else: multi_res_grids = [p.grids for p in model.proposal_networks]
                
        else:
            raise NotImplementedError(self.what)
        total = 0
       
        for grids in multi_res_grids:
            for id, grid in enumerate(grids):
                if (self.what=='field' and self.info['field']) or (self.what=='proposal_network' and self.info['psn']):
                    total += compute_plane_smoothness(grid)
                    # total += compute_plane_smoothness_wavelet(grid)
                else:
                    if id in [2,4,5]:
                        total += compute_plane_smoothness(grid)
        return torch.as_tensor(total)
    
# Original L1TimePlanes class
class L1TimePlanes(Regularizer):
    def __init__(self, initial_value, what='field', info=None):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        super().__init__(f'l1-time-{what[:2]}', initial_value)
        self.what = what
        self.info = info

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids: Sequence[nn.ParameterList]
        
        if self.what == 'field':
            if self.info['field']: multi_res_grids = model.field.grids(time_only=True, regularise_wavelet_coeff=True)
            else: multi_res_grids = model.field.grids

        elif self.what == 'proposal_network':
            if self.info['psn']: 
                multi_res_grids = [p.grids(time_only=True, regularise_wavelet_coeff=True) for p in model.proposal_networks]
                multi_res_grids = multi_res_grids[0] + multi_res_grids[1] 
            else: multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        
        total = 0.0
        for grids in multi_res_grids:         
            # Changes from reg w.r.t each time plane to w.r.t all time planes
            for id, grid in enumerate(grids):
                # Regularisation for ours
                if (self.what=='field' and self.info['field']) or (self.what=='proposal_network' and self.info['psn']):
                    total += torch.pow(grid.abs(), 1).mean()
                else:
                    if id in [2,4,5]:
                        total += torch.pow((1.-grid).abs(), 1).mean()

        return torch.as_tensor(total)

class L1ProposalNetwork(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-proposal-network', initial_value)

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        grids = [p.grids() for p in model.proposal_networks][0] # [p.grids for p in model.proposal_networks]#[p.grids() for p in model.proposal_networks][0]
        total = 0.0
        for pn_grids in grids:
            for grid in pn_grids:
                total += torch.abs(1-grid).mean()
        return torch.as_tensor(total)

class DepthTV(Regularizer):
    def __init__(self, initial_value):
        super().__init__('tv-depth', initial_value)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        depth = model_out['depth']
        tv = compute_plane_tv(
            depth.reshape(64, 64)[None, None, :, :]
        )
        return tv
    
# Not main regularisation
class HistogramLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('histogram-loss', initial_value)

        self.visualize = False
        self.count = 0

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        if self.visualize:
            if self.count % 500 == 0:
                prop_idx = 0
                fine_idx = 1
                # proposal info
                weights_proposal = model_out["weights_list"][prop_idx].detach().cpu().numpy()
                spacing_starts_proposal = model_out["ray_samples_list"][prop_idx].spacing_starts
                spacing_ends_proposal = model_out["ray_samples_list"][prop_idx].spacing_ends
                sdist_proposal = torch.cat([
                    spacing_starts_proposal[..., 0],
                    spacing_ends_proposal[..., -1:, 0]
                ], dim=-1).detach().cpu().numpy()

                # fine info
                weights_fine = model_out["weights_list"][fine_idx].detach().cpu().numpy()
                spacing_starts_fine = model_out["ray_samples_list"][fine_idx].spacing_starts
                spacing_ends_fine = model_out["ray_samples_list"][fine_idx].spacing_ends
                sdist_fine = torch.cat([
                    spacing_starts_fine[..., 0],
                    spacing_ends_fine[..., -1:, 0]
                ], dim=-1).detach().cpu().numpy()

                for i in range(10):  # plot 10 rays
                    fix, ax1 = plt.subplots()

                    delta = np.diff(sdist_proposal[i], axis=-1)
                    ax1.bar(sdist_proposal[i, :-1], weights_proposal[i].squeeze() / delta, width=delta, align="edge", label='proposal', alpha=0.7, color="b")
                    ax1.legend()
                    ax2 = ax1.twinx()

                    delta = np.diff(sdist_fine[i], axis=-1)
                    ax2.bar(sdist_fine[i, :-1], weights_fine[i].squeeze() / delta, width=delta, align="edge", label='fine', alpha=0.3, color='r')
                    ax2.legend()
                    os.makedirs(f'histogram_loss/{self.count}', exist_ok=True)
                    plt.savefig(f'./histogram_loss/{self.count}/batch_{i}.png')
                    plt.close()
                    plt.cla()
                    plt.clf()
            self.count += 1
        return interlevel_loss(model_out['weights_list'], model_out['ray_samples_list'])
    

class DistortionLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('distortion-loss', initial_value)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        """
        Efficient O(N) realization of distortion loss.
        from https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py
        There are B rays each with N sampled points.
        """
        w = model_out['weights_list'][-1]
        rs: RaySamples = model_out['ray_samples_list'][-1]
        m = (rs.starts + rs.ends) / 2
        interval = rs.deltas

        loss_uni = (1/3) * (interval * w.pow(2)).sum(dim=-1).mean()
        wm = (w * m)
        w_cumsum = w.cumsum(dim=-1)
        wm_cumsum = wm.cumsum(dim=-1)
        loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
        loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
        loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
        return loss_bi + loss_uni
