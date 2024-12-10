import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

from pytorch_wavelets_.dwt.transform2d import DWTInverse, DWTForward

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

def interpolate_features_MUL(pts: torch.Tensor, kplanes, idwt, ro_grid):
    """Generate features for each point
    """
    # rot_pts = torch.matmul(pts[..., :3], ro_grid)
    # pts_ = torch.cat([rot_pts, pts[..., -1].unsqueeze(-1)], dim=-1)

    # initialise the feature space
    interp = []
    
    # q,r are the coordinate combinations needed to retrieve pts
    q,r = 0,1
    for i in range(6):
        
        coeff = kplanes[i]
        
        ms_features = coeff(pts[..., (q,r)], idwt) # list returned in order of fine to coarse features
        # Initialise interpolated space
        if interp == []:
            interp = [1. for j in range(len(ms_features))]
        
        for j, feature in enumerate(ms_features):
            interp[j] = interp[j] * feature

        r += 1
        if r == 4:
            q += 1
            r = q+1
   
    # Concatenate ms_features
    ms_interp = torch.cat(interp, dim=-1)

    return ms_interp

def interpolate_features_ZAM(pts: torch.Tensor, kplanes, idwt):
    """Generate features for each point
    """
    interp_sum = []      
    interp = []
    # q,r are the coordinate combinations needed to retrieve pts
    q,r = 0,1
    for i in range(6):
        if i in [2,4,5]:
            coeff = kplanes[i]

            ms_features = coeff(pts[..., (q,r)], idwt) # list returned in order of fine to coarse features            

            # Initialise interpolated space
            if interp == []:
                interp = [1. for j in range(len(ms_features))]
                interp_sum = [0. for j in range(len(ms_features))]

            for j, feature in enumerate(ms_features):
                # Sum features
                interp[j] = interp[j] * feature
                interp_sum[j] = interp_sum[j] + feature

                # On final it of spacetime plane
                if i == 5:
                    # Invert agreement mask so 0. indicates agreed 0-value and 1. indicates not agreed
                    interp_sum[j] = interp_sum[j] / 3.
        r += 1
        if r == 4:
            q += 1
            r = q+1

    # Now deal with space-only features
    q,r = 0,1
    for i in range(6):
        if i in [0,1,3]:
            coeff = kplanes[i]

            ms_features = coeff(pts[..., (q,r)], idwt) # list returned in order of fine to coarse features            

            for j, feature in enumerate(ms_features):
                interp[j] = interp[j] * feature

                if i == 3:
                    interp[j] = interp[j] * interp_sum[j]
        r += 1
        if r == 4:
            q += 1
            r = q+1
    
    # Concatenate ms_features
    ms_interp = torch.cat(interp, dim=-1)
    return ms_interp

# Define the grid
class GridSet(nn.Module):

    def __init__(
            self,
            what: str, # Space/Spacetime  
            resolution: list,   
            config: dict = {},
            is_proposal:bool=False,
            J:int=3,
            cachesig:bool=True,
            ):
        super().__init__()

        self.what = what
        self.is_proposal = is_proposal
        self.running_compressed = False
        self.cachesig = cachesig

        init_mode = 'uniform'
        if self.what == 'spacetime':
            init_mode = 'ones'

        self.feature_size = config['feature_size']
        self.resolution = resolution
        self.wave = config['wave']
        self.mode = config['wave_mode']
        self.J = J

        # Initialise a signal to DWT into our initial Wave coefficients
        dwt = DWTForward(J=J,wave=config['wave'], mode=config['wave_mode']).cuda()
        init_plane = torch.empty(
            [1, config['feature_size'], resolution[0], resolution[1]]
        ).cuda()
        
        if init_mode == 'uniform':
            nn.init.uniform_(init_plane, a=config['a'], b=config['b'])
        elif init_mode == 'zeros':
            nn.init.zeros_(init_plane)
        elif init_mode == 'ones':
            nn.init.ones_(init_plane)
        else:
            raise AttributeError("init_mode not given")

        if self.what == 'spacetime':
            init_plane = init_plane - 1.
        (yl, yh) = dwt(init_plane)
        
        # Initialise coefficients
        grid_set = [nn.Parameter(yl.clone().detach())] +\
            [nn.Parameter(y.clone().detach()) for y in yh]

        coef_scaler = [1., .2, .4, .6, .8]
        grids = []

        for i in range(self.J+1):       
            grids.append((1./coef_scaler[i]) * grid_set[i])

        # Rescale so our initial coeff return initialisation
        self.grids = nn.ParameterList(grids)
        self.scaler = coef_scaler

        del yl, yh, dwt, init_plane, grid_set
        torch.cuda.empty_cache()

        self.step = 0.
        self.signal = 0.

    def compact_save(self):
        """Construct the dictionary containing non-zero coefficient values"""
        # Rescale coefficient values
        coeffs = []
        for i in range(self.J + 1):
            coeffs.append(self.grids[i])

        dictionary = {}
        data = {}

        for i_ in range(self.J + 1):
            cs = coeffs[i_].squeeze(0)
            n = 0.1

            lt = (cs < n)
            gt = (cs > -n)
            cs = ~(lt * gt) * cs  # .nonzero(as_tuple=True)
            nzids = (cs == 0.).nonzero(as_tuple=True)
            non_zero_mask = cs.nonzero()

            cs = cs.tolist()

            i = f'{i_}'
            data[i] = {}
            # Deal with father wavelets first (shape B, H, W)
            if i == '0':
                dictionary = {f'{k}.{l}.{m}': {'val': cs[k][l][m], 'l': f'{l}', 'm': f'{m}'} for (k, l, m) in
                              non_zero_mask.tolist()}

                # # Reformat
                for k_ in dictionary.keys():
                    k = k_.split('.')[0]
                    # Get branch keys
                    l = dictionary[k_]['l']
                    m = dictionary[k_]['m']
                    val = dictionary[k_]['val']

                    # Construct k branch
                    if k not in data[i].keys():
                        data[i][k] = {}
                    # Construct l branch
                    if l not in data[i][k].keys():
                        data[i][k][l] = {}

                    data[i][k][l][m] = val
            else:  # Deal with mother wavelets (shape B, F, H, W)
                dictionary = {f'{k}.{n}.{l}.{m}': {'val': cs[k][n][l][m], 'n': f'{n}', 'l': f'{l}', 'm': f'{m}'} for
                              (k, n, l, m) in non_zero_mask.tolist()}

                # Reformat
                for k_ in dictionary.keys():
                    k = k_.split('.')[0]
                    # Get branch keys
                    l = dictionary[k_]['l']
                    m = dictionary[k_]['m']
                    n = dictionary[k_]['n']
                    val = dictionary[k_]['val']

                    # Construct k branch
                    if k not in data[i].keys():
                        data[i][k] = {}
                    # Construct l branch
                    if n not in data[i][k].keys():
                        data[i][k][n] = {}
                    # Construct l branch
                    if l not in data[i][k][n].keys():
                        data[i][k][n][l] = {}

                    if m in data[i][k][n][l].keys():
                        print('Index already defined: likely defined previously by father wavelet')

                    data[i][k][n][l][m] = val

        return data

    def wave_coefs(self, notflat:bool=False):
        # Rescale coefficient values
        ms = []
        for i in range(self.J+1):
            if i == 0:
                ms.append(self.scaler[i]*self.grids[i])
            else:
                
                co = self.scaler[i]*self.grids[i]
                
                # Flatten / Dont flatten
                if notflat:
                    ms.append(co)
                else:
                    ms.append(co.flatten(1,2))
        
        return ms

    def idwt_transform(self, idwt):
        coeffs = []
        for i in range(self.J+1):
            coeffs.append(self.grids[i])

        yl = 0.
        yh = []
        
        for i in range(self.J+1):
            if i == 0:
                yl = self.scaler[i]*coeffs[i]
            else:
                co = self.scaler[i]*coeffs[i]
                yh.append(co)


        fine = idwt((yl, yh))
        coarse = idwt((yl, yh[1:]))

        if self.what == 'spacetime':
            # return [torch.ones_like(fine+1), torch.ones_like(coarse+1.)]
            return [fine+1., coarse+1.]
        return [fine, coarse]
        

    def forward(self, pts, idwt):
        """Given a set of points sample the dwt transformed Kplanes and return features
        """
        # List: coarse to fine with Feature size (1, N, H, W)
        ms_plane = self.idwt_transform(idwt) 
        ms_features = []
        signal = []
        for plane in ms_plane:
            if self.cachesig:
                signal.append(plane.clone())
            
            if self.what == 'spacetime':
                # Sample features
                feature = (
                        grid_sample_wrapper(plane, pts, st=True)
                        .view(-1, plane.shape[1])
                    )
            else:
                # Sample features
                feature = (
                        grid_sample_wrapper(plane, pts)
                        .view(-1, plane.shape[1])
                    )
            ms_features.append(feature) 
        
        self.signal = signal
        self.step += 1
        # Return multiscale features
        return ms_features


class HexPlaneField(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True



        # 1. Init planes
        self.grids = nn.ModuleList()

        self.idwt = DWTInverse(wave='coif4', mode='periodization').cuda().float()

        self.cacheplanes = True
        
        res_multiplier = 1
        j,k = 0,1
        for i in range(6):
            if k == 3:
                what = 'spacetime'
                res = [self.grid_config[0]['resolution'][j]*res_multiplier, self.grid_config[0]['resolution'][k]]

                if self.is_static:
                    res = [1, 1]
            else:
                what = 'space'
                res = [self.grid_config[0]['resolution'][j]*res_multiplier, self.grid_config[0]['resolution'][k]*res_multiplier]
            
            gridset = GridSet(
                what=what, 
                resolution=res, 
                J=2,
                config={
                    'feature_size': self.grid_config[0]["output_coordinate_dim"],
                    'a':0.1,
                    'b':0.5,
                    'wave':'coif4',
                    'wave_mode':'periodization',
                },
                cachesig=self.cacheplanes
            )

            self.grids.append(gridset)
        
        self.feat_dim = self.grid_config[0]["output_coordinate_dim"] * 2

        init_plane = torch.empty([3,3]).cuda()
        nn.init.uniform_(init_plane, a=-0.1, b=.1)
        self.reorient_grid = nn.Parameter(init_plane, requires_grad=True)

        # self.compact_save()

    def compact_save(self, fp):
        import lzma
        import pickle
        data = {}

        for i in range(6):
            data[f'{i}'] = self.grids[i].compact_save()

        with lzma.open(f"{fp}.xz", "wb") as f:
            pickle.dump(data, f)

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)
    
    def grids_(self, regularise_wavelet_coeff:bool=False, time_only:bool=False, notflat:bool=False):
        """Return the grids as a list of parameters for regularisation
        """
        if regularise_wavelet_coeff:
            # Retrive coefficients to regularise
            ms_planes = []
            for i in range(6):
                # Skip space planes in time only
                if time_only and i not in [2,4,5]:
                    continue

                gridset = self.grids[i]

                ms_feature_planes = gridset.wave_coefs(notflat=notflat)

                if ms_planes == []:
                    ms_planes = [[] for j in range(len(ms_feature_planes))]
                
                for j, feature_plane in enumerate(ms_feature_planes):
                    ms_planes[j].append(feature_plane)
                
        else:
            # Retrive planes to regularise
            ms_planes = []
            for i in range(6):
                if time_only and i not in [2,4,5]:
                    continue
                
                gridset = self.grids[i]

                if self.cacheplanes:
                    ms_feature_planes = gridset.signal
                else:
                    ms_feature_planes = gridset.idwt_transform(self.idwt)

                # Initialise empty ms_planes
                if ms_planes == []:
                    ms_planes = [[] for j in range(len(ms_feature_planes))]
                
                for j, feature_plane in enumerate(ms_feature_planes):
                    ms_planes[j].append(feature_plane)
                
        return ms_planes
    
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_features_MUL(
            pts, self.grids, self.idwt, self.reorient_grid
            )

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
