import torch
import torch.nn as nn
import tinycudann as tcnn
from PIL import Image
import matplotlib.pyplot as plt

from plenoxels.ops.interpolation import grid_sample_wrapper
from pytorch_wavelets_.dwt.transform2d import DWTForward

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


""" The following functions are used to implement the different feature fusion schemes
"""
def interpolate_features_MUL(pts: torch.Tensor, kplanes, idwt, is_static):
    """Generate features for each point
    """
    # initialise the feature space
    interp = []
    
    
    # q,r are the coordinate combinations needed to retrieve pts
    q,r = 0,1
    static_tic = 0
    for i in range(6):
        
        # Skip Spacetime Planes
        if is_static and i in [2,4,5]:
            continue
        
        coeff = kplanes[i]
        
        ms_features = coeff(pts[..., (q,r)], idwt) # list returned in order of fine to coarse features            
        # Initialise interpolated space
        if interp == []:
            interp = [1. for j in range(len(ms_features))]
        
        for j, feature in enumerate(ms_features):
            interp[j] = interp[j] * feature
        
        # If static scene we need the pts indices to be 01, 02, 12
        if is_static:
            r += 1
            if r == 3:
                q += 1
                r = q+1

        # Otherwise if dynamic we need pts indices to be 01, 02, 03, 12, 13, 23
        else:
            r += 1
            if r == 4:
                q += 1
                r = q+1
   
    # Concatenate ms_features
    ms_interp = torch.cat(interp, dim=-1)

    return ms_interp

def interpolate_features_ADD(pts: torch.Tensor, kplanes, idwt):
    """Generate features for each point
    """
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
            interp[j] = interp[j] + feature
        
        r += 1
        if r == 4:
            q += 1
            r = q+1
    
    # Concatenate ms_features
    ms_interp = torch.cat(interp, dim=-1)

    return ms_interp

def interpolate_features_ZAM_old(pts: torch.Tensor, kplanes, idwt):
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

def interpolate_features_ZAM(pts: torch.Tensor, kplanes, idwt):
    """Generate features for each point
    """
    # initialise the feature space
    interp = []
    q,r = 0,1
    sp_coeffs = {
        'yl':[],
        'yh':[]
    }
    st_coeffs = {
        'yl':[],
        'yh':[]
    }

    # Retrieve coefficients and organise
    for i in range(6):
        op = kplanes[i]()
        if i in [0,1,3]:# space-only
            sp_coeffs['yl'].append(op[0])
            for j, val in enumerate(op[1:]):
                if not (str(j) in sp_coeffs.keys()):
                    sp_coeffs[str(j)] = []
                sp_coeffs[str(j)].append(val)
        else:
            st_coeffs['yl'].append(op[0])
            for j, val in enumerate(op[1:]):
                if not (str(j) in st_coeffs.keys()):
                    st_coeffs[str(j)] = []
                st_coeffs[str(j)].append(val)   
        r += 1
        if r == 4:
            q += 1
            r = q+1
    
    # Fuse the wavelet features by finding the higher feature
    for i, key in enumerate(sp_coeffs.keys()):
        if key != 'yh': 
            y_sp, indices = torch.max(torch.cat(sp_coeffs[key], dim=0), dim=0)
            y_st, _ = torch.max(torch.cat(st_coeffs[key], dim=0), dim=0)
            
            print(indices.shape)
            y_sp = y_sp.unsqueeze(0) * kplanes[0].scaler[i]
            y_st = y_st.unsqueeze(0) * kplanes[0].scaler[i]
            if key != 'yl':
                sp_coeffs['yh'].append(y_sp)
                st_coeffs['yh'].append(y_st)
            else:
                sp_coeffs['yl'] = y_sp
                st_coeffs['yl'] = y_st
    
    sp_fine = idwt((sp_coeffs['yl'], sp_coeffs['yh']))
    st_fine = idwt((st_coeffs['yl'], st_coeffs['yh'])) + 1.
    print(sp_fine.shape)
    exit()
    sp_feature = (
                grid_sample_wrapper(sp_fine, pts)
                .view(-1, plane.shape[1])
        )
    st_feature = (
                grid_sample_wrapper(st_fine, pts)
                .view(-1, plane.shape[1])
        )

    features = sp * st

    print(features.shape)
    exit()
    # q,r are the coordinate combinations needed to retrieve pts
    q,r = 0,1
    static_tic = 0
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



class GridSet(nn.Module):

    def __init__(
            self,
            what: str, 
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

        coef_scaler =  [1., 0.4, 0.6, 0.8] #[1., .2, .4, .6, .8]

        grids = []

        for i in range(self.J+1):       
            grids.append((1./coef_scaler[i]) * grid_set[i])

        # Rescale so our initial coeff return initialisation
        self.grids = nn.ParameterList(grids)
        self.scaler = coef_scaler

        # Clean up memory
        del yl, yh, dwt, init_plane, grid_set
        torch.cuda.empty_cache()

        self.step = 0.
        self.signal = 0.
    
    def wave_coefs(self, notflat:bool=False):
        """ Return raw multi-scale wavelet coefficients

            Input:
                notflat - choose to flatten filter axis for mother wavelet coefficients
        """
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

    def compact_save(self,):
        """Construct the dictionary containing non-zero coefficient values
        """
        # Rescale coefficient values
        coeffs = []
        for i in range(self.J+1):
            coeffs.append(self.grids[i])

        dictionary = {}
        data = {}
        
        for i_ in range(self.J+1):
            cs = coeffs[i_].squeeze(0)
            n = 0.1

            lt = (cs < n )
            gt = (cs > -n )
            cs = ~(lt*gt) * cs #.nonzero(as_tuple=True)
            nzids = (cs == 0.).nonzero(as_tuple=True)
            non_zero_mask = cs.nonzero()
            
            cs = cs.tolist()
            
            i = f'{i_}'
            data[i] = {}
            # Deal with father wavelets first (shape B, H, W)
            if i == '0':
                dictionary = {f'{k}.{l}.{m}':{ 'val': cs[k][l][m], 'l':f'{l}', 'm':f'{m}'} for (k, l, m) in non_zero_mask.tolist()}

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
            else: # Deal with mother wavelets (shape B, F, H, W)
                dictionary= {f'{k}.{n}.{l}.{m}': {'val': cs[k][n][l][m],  'n':f'{n}', 'l':f'{l}', 'm':f'{m}'} for (k, n,l, m) in non_zero_mask.tolist()}
                
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

    def compact_load(self, dictionary):
        """Load the dictionary containing non-zero wavelet coefficients
        """
        # Rescale coefficient values
        coeffs = []
        for i in range(self.J+1):
            coeffs.append(self.grids[i])
        
        # For each scale
        for i in range(self.J+1):
            # Re-Initialise grid value to 0
            nn.init.zeros_(self.grids[i])

            # Gather data for given plane
            data = dictionary[f'{i}']

            # For the father wavelet we only have 2 branches before reaching our value
            if i == 0:
                # Retrieve values from each hashmap
                for feature_key in data.keys():
                    for axis1_key in data[feature_key].keys():
                        for axis2_key in data[feature_key][axis1_key].keys():
                            self.grids[i][0][int(feature_key)][int(axis1_key)][int(axis2_key)] = data[feature_key][axis1_key][axis2_key]
            
            else: # For mother wavelet we have 3 branches to follow before reaching the value
                # Retrieve values from each hashmap
                for feature_key in data.keys():
                    for filter_key in data[feature_key].keys():
                        for axis1_key in data[feature_key][filter_key].keys():
                            for axis2_key in data[feature_key][filter_key][axis1_key].keys():
                                self.grids[i][0][int(feature_key)][int(filter_key)][int(axis1_key)][int(axis2_key)] = data[feature_key][filter_key][axis1_key][axis2_key]

    def idwt_transform(self, idwt):
        """ Perform the inverse discrete wavelet transfrom and return the feature planes
        """
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

        ms_feats = [fine]

        if not self.is_proposal:
            coarse = idwt((yl, yh[1:]))
            ms_feats.append(coarse)

        if self.what == 'spacetime':
            ms_feats_ = []
            for feat in ms_feats:
                ms_feats_.append(feat + 1.) 
            ms_feats = ms_feats_  
        
        return ms_feats
        
    def forward_(self, pts, idwt):
        """Given a set of points sample the dwt transformed Kplanes and return features
        """
        # List: coarse to fine with Feature size (1, N, H, W)
        ms_plane = self.idwt_transform(idwt) 
        ms_features = []
        signal = []
        for plane in ms_plane:
            if self.cachesig:
                signal.append(plane.clone())
            
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
    
    def forward(self):
        """Given a set of points sample the dwt transformed Kplanes and return features
        """
        self.step += 1

        return self.grids

def plt_grid(grid):
    plotgrid = grid.mean(1).clone().detach().cpu().numpy()
    plt.imshow(plotgrid.squeeze(0))
    plt.show()