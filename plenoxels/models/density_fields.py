"""
Density proposal field
"""
from typing import Optional, Callable
import logging as log

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.raymarching.spatial_distortions import SpatialDistortion

from plenoxels.models.grids import GridSet, interpolate_features_MUL, interpolate_features_ADD, interpolate_features_ZAM, normalize_aabb
from pytorch_wavelets_.dwt.transform2d import DWTInverse, DWTForward


class KPlaneDensityField(nn.Module):
    def __init__(self,
                 aabb,
                 resolution,
                 num_input_coords,
                 num_output_coords,
                 density_activation: Callable,
                 spatial_distortion: Optional[SpatialDistortion] = None,
                 linear_decoder: bool = True,
                 regularise_wavelet_coeff:bool=False,
                 concat_features_across_scales:bool=True,
                 wave = None, wave_mode = None,
                 wave_level:list=[],
                 fusion:str='None',
                 is_static:bool=False,
                 cache_planes:bool=True
                 ):
        super().__init__()

        # Make sure we have a level of decomposition
        assert wave_level != [], AssertionError('Density wave_level is empty: []')

        self.is_static = is_static

        self.fusion_scheme = fusion
        self.cacheplanes = cache_planes


        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        
        self.hexplane = num_input_coords == 4
        self.feature_dim = 2 # num_output_coords * 2
        print(self.feature_dim)
        self.density_activation = density_activation
        
        self.linear_decoder = linear_decoder
        activation = "ReLU"
        if self.linear_decoder:
            activation = "None"

        grid_config = [{
            'resolution':resolution,
            'feature_size': num_output_coords, #num_output_coords,
            'a':0.1,
            'b':0.15,
            'wave': wave,
            'wave_mode': wave_mode,
            'show_planes_before_run':False,
        }]
        self.concat_features_across_scales = concat_features_across_scales
        self.regularise_wavelet_coeff = regularise_wavelet_coeff
        self.wave_level = wave_level


        self.kplanes = nn.ModuleList()
        self.idwt = self.idwt = DWTInverse(wave=grid_config[0]['wave'], mode=grid_config[0]['wave_mode'])

        j,k = 0,1
        for i in range(6):
            what = 'space'
            if k == 3: # when time axis is chosen
                what = 'spacetime'
            
            # Get resolution at position j, k in grid config -> found in config file
            res = [grid_config[0]['resolution'][j], grid_config[0]['resolution'][k]]
            gridset = GridSet(what=what, resolution=res, 
                              config={
                                  'feature_size':grid_config[0]['feature_size'],
                                  'a':0.1,
                                  'b':0.15,
                                  'wave':grid_config[0]['wave'],
                                  'wave_mode':grid_config[0]['wave_mode'],
                                  'show_planes_before_run':grid_config[0]['show_planes_before_run']
                              },
                              is_proposal=True,
                              J=self.wave_level,
                              cachesig=self.cacheplanes
                              )
            
            self.kplanes.append(gridset)

            # Change j,k for next round
            k += 1
            if k == 4:
                j += 1
                k = j + 1

        self.sigma_net = tcnn.Network(
            n_input_dims= self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        ).cuda()

    def compact_save(self, psn_num):
        fp = './fields/'
        data = {}
        for i in range(6):
            if self.is_static:
                if i in [0,1,3]:
                    data[f'{i}'] = self.kplanes[i].compact_save()
            else:
                data[f'{i}'] = self.kplanes[i].compact_save()

        # Compress
        import pickle
        if self.compression_type == 'pickle':
            with open(f'psn_{fp}pickle_field.pickle', 'wb') as f:
                pickle.dump(data, f)

        elif self.compression_type == 'GZIP':
            import gzip
            with gzip.open(f"psn_{fp}gzip_field.gz", "wb") as f:
                pickle.dump(data, f)

        elif self.compression_type == 'BZ2':
            import bz2
            with bz2.BZ2File(f'psn_{fp}bz2_field.pbz2', 'wb') as f:
                pickle.dump(data, f)

        elif self.compression_type == 'LZMA':
            import lzma
            with lzma.open(f"psn_{fp}lzma_field.xz", "wb") as f:
                pickle.dump(data, f)

    def compact_load(self, psn_num):
        """Load compressed model
        """
        fp = './fields/'
        import pickle
        if self.compression_type == 'pickle':
            with open(f'psn_{fp}pickle_field.pickle', 'rb') as handle:
                dictionary = pickle.load(handle)
        elif self.compression_type == 'GZIP':
            import gzip
            with gzip.open(f"psn_{fp}gzip_field.gz", "rb") as f:
                dictionary = pickle.load(f)

        elif self.compression_type == 'BZ2':
            import bz2
            with bz2.BZ2File(f'psn_{fp}bz2_field.pbz2', 'rb') as f:
                dictionary = pickle.load(f)
        
        elif self.compression_type == 'LZMA':
            import lzma
            with lzma.open(f"psn_{fp}lzma_field.xz", "rb") as f:
                dictionary = pickle.load(f)
        
        print(f'Loading Grids ...')
        from tqdm import tqdm
        for i in tqdm(range(6)):
            if self.is_static:
                if i in [0,1,3]:
                    self.kplanes[i].compact_load(dictionary[f'{i}'])
            else:
                self.kplanes[i].compact_load(dictionary[f'{i}'])

    def grids(self, regularise_wavelet_coeff=False, time_only:bool=False, notflat:bool=False):
        """Return the grids as a list of parameters for regularisation
        """
        if self.regularise_wavelet_coeff or regularise_wavelet_coeff:
            # Retrive coefficients to regularise
            ms_planes = []
            for i in range(6):
                if time_only and i not in [2,4,5]:
                    continue
                
                # Skip time planes if we have static scene
                if self.is_static and i in [2,4,5]:
                    continue

                gridset = self.kplanes[i]
            
                ms_feature_planes = gridset.wave_coefs(notflat)

                # Initialise empty ms_planes
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
                    
                # Skip time planes if we have static scene
                if self.is_static and i in [2,4,5]:
                    continue

                gridset = self.kplanes[i]

                if self.cacheplanes:
                    ms_features = gridset.signal
                else:
                    ms_features = gridset.idwt_transform(self.idwt)

                # Initialise empty ms_planes
                if ms_planes == []:
                    ms_planes = [[] for j in range(len(ms_features))]
                for j, feature in enumerate(ms_features):
                    ms_planes[j].append(feature)
                
        return ms_planes
        
    
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]

        if timestamps is not None and self.hexplane:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])

        # Select the feature fusion scheme
        if self.fusion_scheme == 'MUL':
            features = interpolate_features_MUL(pts, self.kplanes, self.idwt, self.is_static)
        elif self.fusion_scheme == 'ADD' and self.is_static == False:
            features = interpolate_features_ADD(pts, self.kplanes, self.idwt)
        elif self.fusion_scheme == 'ZAM' and self.is_static == False:
            features = interpolate_features_ZAM(pts, self.kplanes, self.idwt)
        else:
            raise AttributeError(f"PSN Field Error: No/Incorrect feature fusion scheme: {self.fusion_scheme}")

        density = self.density_activation(
            self.sigma_net(features).to(pts)
        ).view(n_rays, n_samples, 1)

        return density

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)

    def get_params(self):
        field_params = {}
        names = []
        for id, plane in enumerate(self.kplanes):
            for k,v in plane.grids.named_parameters(prefix="grids"):
                field_params[k+f'.{id}'] = v
                names.append(k)

        nn_params = {k: v for k, v in self.sigma_net.named_parameters(prefix="sigma_net")}
        
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys() and 'kplanes' not in k
        )}
        
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
