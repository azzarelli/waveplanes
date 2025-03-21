import torch
import torch.nn as nn

from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn

from pytorch_wavelets_.dwt.transform2d import DWTInverse, DWTForward


from plenoxels.raymarching.spatial_distortions import SpatialDistortion

from plenoxels.models.grids import GridSet, interpolate_features_MUL, interpolate_features_ADD,interpolate_features_ZMM, interpolate_features_ZAM, normalize_aabb, interpolate_features_MUL_LR


class WaveletField(nn.Module):
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        wave_level: Optional[int],
        use_appearance_embedding: bool,
        appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        num_images: Optional[int],
        fusion:str='None',
        is_static:bool=False,
        cache_planes:bool=True
    ) -> None:
        super().__init__()

        # Grid configs:
        self.grid_config = grid_config
        self.fusion_scheme = fusion
        self.wave_level = wave_level
        self.regularise_wavelet_coeff =  grid_config[0]['regularise_wavelet_coeff'] 

        # Select whether feature planes are cached to GPU for faster training or if we reperform IDWT during regularisation
        self.cacheplanes = cache_planes

        # For modelling 3-D scenes
        self.is_static = is_static

        # For compressing the representation (performed after training)
        self.compression_type = grid_config[0]['compression'] if 'compression' in grid_config[0].keys() else 'LZMA'

        # Scene & Decoder parameters
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        self.concat_features_across_scales = concat_features_across_scales
        if concat_features_across_scales:
            self.feature_dim = grid_config[0]['feature_size']* 2 #self.wave_level


        # 1. Initialise planes and the IDWT
        self.kplanes = nn.ModuleList()
        self.idwt = DWTInverse(wave=grid_config[0]['wave'], mode=grid_config[0]['wave_mode']).cuda().float()
        
        # Init wavelet coefficients
        res_multiplier = 1
        j,k = 0,1
        for i in range(6):
            if k == 3:
                what = 'spacetime'
                res = [grid_config[0]['resolution'][j]*res_multiplier, grid_config[0]['resolution'][k]]

                if self.is_static:
                    res = [1, 1]
            else:
                what = 'space'
                res = [grid_config[0]['resolution'][j]*res_multiplier, grid_config[0]['resolution'][k]*res_multiplier]
            
            gridset = GridSet(what=what, resolution=res, 
                              J=self.wave_level,
                              config={
                                  'feature_size':grid_config[0]['feature_size'],
                                  'a':0.1,
                                  'b':0.5,
                                  'wave':grid_config[0]['wave'],
                                  'wave_mode':grid_config[0]['wave_mode'],
                                  'show_planes_before_run':grid_config[0]['show_planes_before_run']
                              },
                              cachesig=self.cacheplanes
                              )
            
         
            self.kplanes.append(gridset)

            # Change j,k for next round
            k += 1
            if k == 4:
                j += 1
                k = j + 1

        # 2. Init appearance code-related parameters
        self.use_average_appearance_embedding = True  # for test-time
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images
        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim

            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = nn.Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.appearance_embedding_dim = 0

        # 3. Init decoder params
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        # 3. Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims= 3 + self.appearance_embedding_dim,#self.direction_encoder.n_output_dims,
                n_output_dims=self.feature_dim * 3, # 3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims= self.feature_dim, #self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = (
                    self.direction_encoder.n_output_dims
                    + self.geo_feat_dim
                    + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
    
    def compact_save(self, fp:str='./fields'):
        """Save main field as a compressed hashmap
        """
        
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
            with open(f'{fp}pickle_field.pickle', 'wb') as f:
                pickle.dump(data, f)

        elif self.compression_type == 'GZIP':
            import gzip
            with gzip.open(f"{fp}gzip_field.gz", "wb") as f:
                pickle.dump(data, f)

        elif self.compression_type == 'BZ2':
            import bz2
            with bz2.BZ2File(f'{fp}bz2_field.pbz2', 'wb') as f:
                pickle.dump(data, f)

        elif self.compression_type == 'LZMA':
            import lzma
            with lzma.open(f"{fp}lzma_field.xz", "wb") as f:
                pickle.dump(data, f)

    def compact_load(self,  fp:str='./fields'):
        """Load compressed model
        """
        import pickle
        if self.compression_type == 'pickle':
            with open(f'{fp}pickle_field.pickle', 'rb') as handle:
                dictionary = pickle.load(handle)
        elif self.compression_type == 'GZIP':
            import gzip
            with gzip.open(f"{fp}gzip_field.gz", "rb") as f:
                dictionary = pickle.load(f)

        elif self.compression_type == 'BZ2':
            import bz2
            with bz2.BZ2File(f'{fp}bz2_field.pbz2', 'rb') as f:
                dictionary = pickle.load(f)
        
        elif self.compression_type == 'LZMA':
            import lzma
            with lzma.open(f"{fp}lzma_field.xz", "rb") as f:
                dictionary = pickle.load(f)
        
        print(f'Loading Grids ...')
        from tqdm import tqdm
        for i in tqdm(range(6)):
            if self.is_static:
                if i in [0,1,3]:
                    self.kplanes[i].compact_load(dictionary[f'{i}'])
            else:
                self.kplanes[i].compact_load(dictionary[f'{i}'])
            
    def grids(self, regularise_wavelet_coeff:bool=False, time_only:bool=False, notflat:bool=False):
        """Return the grids as a list of parameters for regularisation
        """
        if self.regularise_wavelet_coeff or regularise_wavelet_coeff:
            # Retrive coefficients to regularise
            ms_planes = []
            for i in range(6):
                # Skip space planes in time only
                if time_only and i not in [2,4,5]:
                    continue
                
                # Skip time planes if we have static scene
                if self.is_static and i in [2,4,5]:
                    continue

                gridset = self.kplanes[i]

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

                # Skip time planes if we have static scene
                if self.is_static and i in [2,4,5]:
                    continue
                
                gridset = self.kplanes[i]

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


    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None, LR_plane:bool=False):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)

       
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        
        pts = pts.reshape(-1, pts.shape[-1])

        # Select the feature fusion scheme
        if LR_plane:
            features = interpolate_features_MUL_LR(pts, self.kplanes, self.idwt, self.is_static)
        elif self.fusion_scheme == 'MUL':
            features = interpolate_features_MUL(pts, self.kplanes, self.idwt, self.is_static)
        elif self.fusion_scheme == 'ADD' and self.is_static == False:
            features = interpolate_features_ADD(pts, self.kplanes, self.idwt)
        elif self.fusion_scheme == 'ZMM' and self.is_static == False:
            features = interpolate_features_ZMM(pts, self.kplanes, self.idwt)
        elif self.fusion_scheme == 'ZAM' and self.is_static == False:
            features = interpolate_features_ZAM(pts, self.kplanes, self.idwt)
        else:
            raise AttributeError(f"Fiel Error: No/Incorrect feature fusion scheme: {self.fusion_scheme}")

        
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(
                features, [self.geo_feat_dim, 1], dim=-1)

        density = self.density_activation(
            density_before_activation.to(pts)
        ).view(n_rays, n_samples, 1)
        return density, features

    def forward(self,
                pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                LR_plane:bool=False
                ):
        camera_indices = None
        if self.use_appearance_embedding:
            if timestamps is None:
                raise AttributeError("timestamps (appearance-ids) are not provided.")
            camera_indices = timestamps
            timestamps = None
        density, features = self.get_density(pts, timestamps, LR_plane)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)

        if self.linear_decoder:
            color_features = [features]

        if self.use_appearance_embedding:
            if camera_indices.dtype == torch.float32:
                # Interpolate between two embeddings. Currently they are hardcoded below.
                #emb1_idx, emb2_idx = 100, 121  # trevi
                emb1_idx, emb2_idx = 11, 142  # sacre
                emb_fn = self.appearance_embedding
                emb1 = emb_fn(torch.full_like(camera_indices, emb1_idx, dtype=torch.long))
                emb1 = emb1.view(emb1.shape[0], emb1.shape[2])
                emb2 = emb_fn(torch.full_like(camera_indices, emb2_idx, dtype=torch.long))
                emb2 = emb2.view(emb2.shape[0], emb2.shape[2])
                embedded_appearance = torch.lerp(emb1, emb2, camera_indices)
            elif self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                if hasattr(self, "test_appearance_embedding"):
                    embedded_appearance = self.test_appearance_embedding(camera_indices)
                elif self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.appearance_embedding.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = embedded_appearance.view(-1, 1, ea_dim).expand(n_rays, n_samples, -1).reshape(-1, ea_dim)
            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)

        if self.linear_decoder:
            if self.use_appearance_embedding:
                basis_values = self.color_basis(torch.cat([directions, embedded_appearance], dim=-1))
            else:
                basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        return {"rgb": rgb, "density": density}



    def get_params(self):
        field_params = {}
        names = []
        for id, plane in enumerate(self.kplanes):
            for k,v in plane.grids.named_parameters(prefix="grids"):
                field_params[k+f'.{id}'] = v
                names.append(k)

        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and 'kplanes' not in k
        )}
        
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }


        