import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any, List

import pandas as pd
import torch
import torch.utils.data

from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.utils.ema import EMA
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.models.lowrank_model import LowrankModel
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss
)
import wandb
import numpy as np


class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        self.ist_step = ist_step
        self.isg_step = isg_step
        
        self.save_video = save_outputs
        # Switch to compute extra video metrics (FLIP, JOD)
        self.compute_video_metrics = False
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            **kwargs)


    def display_psn0_histogram(self):
        import matplotlib.pyplot as plt

        num_rows = 6
        num_cols = 4

        # Create a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots

        field = self.model.proposal_networks[0].kplanes
        for j, plane in enumerate(field):
            row = j
            if plane.what == 'spacetime':
                for i, grid in enumerate(plane.grids):

                    
                    flat_grid = grid.flatten().cpu().numpy()

                    col = i % num_cols
                    flattened_matrix = flat_grid.flatten()  # Flatten the 3-D matrix to a 1-D array
                    
                    if i == 1:
                        flattened_matrix = flattened_matrix*4.

                    axes[row, col].hist(flattened_matrix, bins=40, color='blue', alpha=0.5)

                    if row == 0:
                        axes[row, col].set_title(f'Lvl: {i}')
                        axes[row, col].grid(True)
            
        for j, plane in enumerate(field):
            row = j
            col = 3
            if plane.what == 'spacetime':
                ms_feats = []
                for feature in plane.idwt_transform(self.model.proposal_networks[0].idwt):
                    ms_feats.append(feature.flatten().unsqueeze(0))
                flattened_matrix = torch.cat(ms_feats,  dim=-1).squeeze(0).cpu().numpy()

                axes[row, col].hist(flattened_matrix, bins=40, color='red', alpha=0.5)
                if row == 0:
                    axes[row, col].set_title(f'Signal')
                axes[row, col].grid(True)

        plt.show()

    def display_kplane_histogram(self):
        import matplotlib.pyplot as plt

        num_rows = 6
        num_cols = 4

        # Create a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots


        for j, msplane in enumerate(self.model.field.grids):
            col = j

            for i, grid in enumerate(msplane):
            
                flat_grid = (grid).flatten().cpu().numpy()
     
                row = i 

                color = 'blue'
             
                flattened_matrix = flat_grid.flatten()  # Flatten the 3-D matrix to a 1-D array
                axes[row, col].hist(flattened_matrix, bins=40, color=color, alpha=0.5)

                if row == 0:
                    axes[row, col].set_title(f'Scale {i}')
                    axes[row, col].grid(True)
                if col ==0:
                    axes[row, col].set_ylabel(f'id {j}')
  
        plt.show()

    def display_waveplane_histogram(self):
        import matplotlib.pyplot as plt

        num_rows = 6
        num_cols = 5

        # Create a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots


        for j, plane in enumerate(self.model.field.kplanes):
            row = j

            for i, grid in enumerate(plane.grids):

                flat_grid = (plane.scaler[i]*grid).flatten().cpu().numpy()

                col = i % num_cols
                print(plane.scaler[i], col)

                color = 'blue'
                if plane.what == 'spacetime':
                    color='green'
                flattened_matrix = flat_grid.flatten()  # Flatten the 3-D matrix to a 1-D array
                axes[row, col].hist(flattened_matrix, bins=40, color=color, alpha=0.5)

                if row == 0:
                    axes[row, col].set_title(f'Lvl: {i}')
                    axes[row, col].grid(True)
                if col ==0:
                    axes[row, col].set_ylabel(plane.what)
        
        for j, plane in enumerate(self.model.field.kplanes):
            row = j
            col = num_cols-1
            ms_feats = []
            for feature in plane.idwt_transform(self.model.field.idwt):
                ms_feats.append(feature.flatten().unsqueeze(0))
            flattened_matrix = torch.cat(ms_feats,  dim=-1).squeeze(0).cpu().numpy()

            axes[row, col].hist(flattened_matrix, bins=40, color='red', alpha=0.5)
            if row == 0:
                axes[row, col].set_title(f'Signal')
            
                # axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True)

            # print(ms_features.shape)
        plt.show()


    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)

        # Hard-coded: Uncomment to analyse the desired planes using histograms
        # self.display_kplane_histogram()
        # self.display_waveplane_histogram()
        # self.display_psn0_histogram()

        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)

                # We want to deform the model during tests
                outputs = self.model(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far
                    )
                
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        scale_ok = super().train_step(data, **kwargs)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale_ok

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate(self, is_compressing:bool=False):
        dataset = self.test_dataset
        
        if is_compressing:
            # Hard-coded: compress proposal networks
            # for psn_num, psn in enumerate(self.model.proposal_networks):
            #     print(f'Proposal Net {psn_num}')
            #     psn.compact_save(psn_num)
            #     psn.compact_load(psn_num)
            print('Compressing model ...')
            self.model.field.compact_save(self.log_dir)
            print('Model saved, now loading ...')
            self.model.field.compact_load(self.log_dir)

        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        sum_ = 0.
        sum__ = 0.
        sum_psnr_f = 0. 
        sum_psnr_b = 0.
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data)

            gtimg =  dataset.imgs[img_idx].float() /255.
            out_metrics, out_img, out_depth = self.evaluate_metrics(
                gtimg, preds, dset=dataset, img_idx=img_idx, name=None,
                #data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)

            sum_ += out_metrics['psnr']
            sum__ += out_metrics['ssim']
            sum_psnr_f += out_metrics['psnr-foreground']
            sum_psnr_b += out_metrics['psnr-background']
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)

        if wandb.run is not None:
            wandb.log({'final_psnr':float(sum_/(img_idx+1))})
            wandb.log({'final_ssim':float(sum__/(img_idx+1))})
            wandb.log({'final_psnr_background':float(sum_psnr_b/(img_idx+1))})
            wandb.log({'final_psnr_foreground':float(sum_psnr_f/(img_idx+1))})

        pb.close()
        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                pred_frames
            )
            if len(out_depths) > 0:
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                    out_depths
                )
        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            per_scene_metrics["JOD"] = metrics.jod(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )
            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )

        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field', info=self.model.what),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network', info=self.model.what),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field', info=self.model.what),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network', info=self.model.what),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field', info=self.model.what),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network', info=self.model.what),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Video360Dataset with downsample={data_downsample}")
    tr_dset = Video360Dataset(
        data_dir, split='train', downsample=data_downsample,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
    )
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    g = torch.Generator()
    g.manual_seed(0)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,  prefetch_factor=4, pin_memory=True,
        worker_init_fn=init_dloader_random, generator=g)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, **kwargs):
    downsample = 2.0 # Both D-NeRF and DyNeRF use downsampling by 2

    ts_dset = Video360Dataset(
        data_dir, split=split, downsample=downsample,
        max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
    )

    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, **kwargs))
    return od
