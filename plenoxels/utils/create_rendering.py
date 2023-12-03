"""Entry point for simple renderings, given a trainer and some poses."""
import os
import logging as log
from typing import Union

import torch

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.static_trainer import StaticTrainer
from plenoxels.runners.video_trainer import VideoTrainer


@torch.no_grad()
def render_to_path(trainer: Union[VideoTrainer, StaticTrainer], extra_name: str = "") -> None:
    """Render all poses in the `test_dataset`, saving them to file
    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    dataset = trainer.test_dataset

    # msgrids = trainer.model.field.grids(regularise_wavelet_coeff=True, notflat=True)
    # # psns =  #.grids(regularise_wavelet_coeff=True, notflat=True)
    # psn = []
    # for p in trainer.model.proposal_networks:
    #     psn.append(p.grids(regularise_wavelet_coeff=True, notflat=True))
    
    # # For each set of grids in multiscale grid list
    # for grids in msgrids:
    #     # For each grid in the set of grids
    #     for grid in grids:
    #         print(grid.shape)
    #         print((grid < 0.00001).nonzero(as_tuple=True)[0].shape[0]/(grid).nonzero(as_tuple=True)[0].shape[0])
    pb = tqdm(total=dataset.timestamps.shape[0], desc=f"Rendering scene")
    frames = []
    for img_idx, data in enumerate(dataset):
        ts_render = trainer.eval_step(data)

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
        preds_rgb = (
            ts_render["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .numpy()
        )
        frames.append(preds_rgb)
        pb.update(1)
    pb.close()

    out_fname = os.path.join(trainer.log_dir, f"ST_Ones_ZMM_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")


def normalize_for_disp(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


@torch.no_grad()
def decompose_space_time(trainer: StaticTrainer, extra_name: str = "") -> None:
    """Render space-time decomposition videos for poses in the `test_dataset`.

    The space-only part of the decomposition is obtained by setting the time-planes to 1.
    The time-only part is obtained by simple subtraction of the space-only part from the full
    rendering.

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    chosen_cam_idx = 15
    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset 

    # Store original parameters from main field and proposal-network field
    parameters = []
    for c in model.field.kplanes:
        parameters.append([grid.data for grid in c.grids])

    pn_parameters = []
    for pn in model.proposal_networks:
        pnp = []
        for plane in pn.kplanes:
            pnp.append([grid_plane.data for grid_plane in plane.grids])
        pn_parameters.append(pnp)
        
    camdata = None
    for img_idx, data in enumerate(dataset):
        if img_idx == chosen_cam_idx:
            camdata = data
    if camdata is None:
        raise ValueError(f"Cam idx {chosen_cam_idx} invalid.")

    num_frames = img_idx + 1
    frames = []
    for img_idx in tqdm(range(num_frames), desc="Rendering scene with separate space and time components"):
        # Linearly interpolated timestamp, normalized between -1, 1
        camdata["timestamps"] = torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        # Full model: turn on time-planes
        for i in range(len(model.field.kplanes)):
            if i in [2,4,5]:
                for j in range(len(parameters[i])):
                    model.field.kplanes[i].grids[j].data = parameters[i][j]
        
        for pn in range(len(model.proposal_networks)):
            for i in [2, 4, 5]:
                for j in range(len(pn_parameters[pn][i])):
                    model.proposal_networks[pn].kplanes[i].grids[j].data = pn_parameters[pn][i][j]
        
        preds = trainer.eval_step(camdata)
        full_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Space-only model: turn off time-planes
        for i in range(len(model.field.kplanes)):
            if i in [2,4,5]:
                for j in range(len(parameters[i])):
                    model.field.kplanes[i].grids[j].data = torch.zeros_like(parameters[i][j])
        
        for pn in range(len(model.proposal_networks)):
            for i in [2, 4, 5]:
                for j in range(len(pn_parameters[pn][i])):
                    model.proposal_networks[pn].kplanes[i].grids[j].data = torch.zeros_like(pn_parameters[pn][i][j])

        preds = trainer.eval_step(camdata)
        spatial_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Temporal model: full - space
        temporal_out = normalize_for_disp(full_out - spatial_out)

        frames.append(
            torch.cat([full_out, spatial_out, temporal_out], dim=1)
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"spacetime_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")

import matplotlib.pyplot as plt
@torch.no_grad()
def plane_plot(trainer: StaticTrainer, extra_name: str = "") -> None:
    """

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    idxs = [0,1,3] # [2,4,5]

    model: LowrankModel = trainer.model

    # Store original parameters from main field and proposal-network field
    parameters = []
    for multires_grids in model.field.grids:
        parameters.append([grid.data for grid in multires_grids])

    for idx, grid in enumerate(multires_grids):
        if idx in idxs:            
            # Checkthat grid has data
            if grid.sum()/(grid.shape[0] *grid.shape[1] * grid.shape[2]* grid.shape[3]) != 1.:
                    batch_size, c, h, w = grid.shape

                    first_difference = grid[..., 1:, :] - grid[..., :h-1, :]  # [batch, c, h-1, w]
                    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    
                    grid_ = grid.mean(1).squeeze(0) #first_difference.squeeze(0).mean(0)

                    img_ready = grid_.clamp(0., 1.).mul(255.0).byte().cpu().numpy()

                    plt.imshow(img_ready, cmap='gray')
                    plt.axis('off')  # Turn off axis
                    plt.show()


    log.info(f"Some log message")

def compute_t0_difference(grids):
    # Take xy and and (xt*xy) at t=0 and take difference:
    # I.e. we looks at conditioning the static scene to t=0

    # 0, 1, 3 are the static planes
    # 2, 4, 5 are the dynamic planes
    # Get feature vector at t0 (first row in H-dim for shape (B,C,H,W))

    xt = grids[2][..., 0, :].squeeze(0).unsqueeze(1)
    yt = grids[4][..., 0, :].squeeze(0).unsqueeze(1)
    zt = grids[5][..., 0, :].squeeze(0).unsqueeze(1)
    
    xy = grids[1]
    xz = grids[0]
    yz = grids[3]

    xy_ = torch.bmm(xt.transpose(1,2), yt).unsqueeze(0)

    xz_ = torch.bmm(xt.transpose(1,2), zt).unsqueeze(0)

    yz_ = torch.bmm(yt.transpose(1,2), zt).unsqueeze(0)

    return (xy, xy_), (xz, xz_), (yz, yz_)

import matplotlib.pyplot as plt
@torch.no_grad()
def mul_xy_xt_zt(trainer: StaticTrainer, extra_name: str = "") -> None:
    """

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    model: LowrankModel = trainer.model

    # Store original parameters from main field and proposal-network field
    parameters = []
    for multires_grids in model.field.grids:
        parameters.append(multires_grids)

    for idx, grid in enumerate(parameters):
        XY, XZ, YZ = compute_t0_difference(grid)
        (xy, xy_), (xz, xz_), (yz, yz_) = XY, XZ, YZ
        print((xy-xy_).mean(), (xz-xz_).mean(), (yz-yz_).mean())

        # if idx in [2,4,5]:            
        #     # Checkthat grid has data
        #     if grid.sum()/(grid.shape[0] *grid.shape[1] * grid.shape[2]* grid.shape[3]) != 1.:
        #             batch_size, c, h, w = grid.shape

        #             first_difference = grid[..., 1:, :] - grid[..., :h-1, :]  # [batch, c, h-1, w]
        #             second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    
        #             grid_ = first_difference.squeeze(0).mean(0)

        #             img_ready = grid_.clamp(0., 1.).mul(255.0).byte().cpu().numpy()

        #             plt.imshow(img_ready, cmap='gray')
        #             plt.axis('off')  # Turn off axis
        #             plt.show()


    log.info(f"Some log message")