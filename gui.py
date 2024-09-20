import os
import time
import torch
import math
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

# from scene.cameras import MiniCam


import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from pytorch_msssim import ms_ssim

from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list

from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from pytorch_msssim import ms_ssim

from utils.scene_utils import render_training_image
from time import time
import copy
from gaussian_renderer import render, network_gui


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from train import scene_reconstruction

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, time):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.c2w = c2w
        self.time = time
        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        # self.rot = R.from_matrix(np.eye(3))
        self.rot = R.from_matrix(np.array([[1., 0., 0.,],
                                           [0., 0., -1.],
                                           [0., 1., 0.]]))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.side = np.array([1, 0, 0], dtype=np.float32)

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        up = self.rot.as_matrix()[:3, 1]
        rotvec_x = up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

    def get_proj_matrix(self):
        tanHalfFovY = math.tan((self.fovy  / 2))
        tanHalfFovX = math.tan((self.fovx / 2))

        top = tanHalfFovY * self.near
        bottom = -top
        right = tanHalfFovX * self.near
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * self.near / (right - left)
        P[1, 1] = 2.0 * self.near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.far / (self.far - self.near)
        P[2, 3] = -(self.far * self.near) / (self.far - self.near)
        return P


class GUI:
    def train_coarse(self):
        scene_reconstruction(
                self.dataset, 
                self.opt, 
                self.hyperparams, 
                self.pipe, 
                self.testing_iterations, 
                self.saving_iterations,
                self.checkpoint_iterations,
                self.checkpoint,
                self.debug_from,
                self.gaussians, 
                self.scene, 
                "coarse", 
                self.tb_writer, 
                self.opt.coarse_iterations,
                self.timer)
    
    def init_taining(self):
        first_iter = 1

        # Set up gaussian training
        self.gaussians.training_setup(self.opt)

        # Load from fine model if it exists
        if self.checkpoint:
            if 'fine' in self.checkpoint: 
                (model_params, first_iter) = torch.load(self.checkpoint)
                self.gaussians.restore(model_params, self.opt)

        # Set current iteration
        self.iteration = first_iter

        # Define background
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Events for counting duration of step
        self.iter_start = torch.cuda.Event(enable_timing = True)
        self.iter_end = torch.cuda.Event(enable_timing = True)
        
        # 
        viewpoint_stack = None
        
        # Define loss and psnr values for logging
        self.ema_loss_for_log = 0.0
        self.ema_psnr_for_log = 0.0

        # Define final iteration of training
        self.final_iter = self.opt.iterations

        # Get the test and training cameras (contain the dataset via `.dataset[#]`)
        # video_cams = self.scene.getVideoCameras()
        self.test_cams = self.scene.getTestCameras()
        self.train_cams = self.scene.getTrainCameras()

        # If the `viewpoint_stack` variable is tbd and the optimiser has not been loaded
        if not viewpoint_stack and not self.opt.dataloader:
            # dnerf's branch
            viewpoint_stack = [i for i in self.train_cams]
            self.temp_list = copy.deepcopy(viewpoint_stack)

        self.test_viewpoint_stack = [i for i in self.test_cams]
        self.test_temp_list = copy.deepcopy(self.test_viewpoint_stack)


        batch_size = self.opt.batch_size
        print("Fine-grained training data loading done")
        
        # If the data loader has been defined, stack training data for data loading
        if self.opt.dataloader:
            viewpoint_stack = self.scene.getTrainCameras()
            if self.opt.custom_sampler is not None:
                print('Custom sampler loaded')
                sampler = FineSampler(viewpoint_stack)
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
                random_loader = False
            else:
                print('Using original 4DGS sampler')
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
                random_loader = True
            loader = iter(viewpoint_stack_loader)
        else:
            print('Data loader not loaded')
            random_loader = None
            loader = None
        # else:
        #     print('Data loader not loaded')
        #     random_loader = False
        #     loader = None

        self.loader = loader
        self.random_loader = random_loader
        self.viewpoint_stack = viewpoint_stack
        self.viewpoint_stack_loader = viewpoint_stack_loader

        self.load_in_memory = False 
        
    def __init__(self, 
                 args, 
                 hyperparams, 
                 dataset, 
                 opt, 
                 pipe, 
                 testing_iterations, 
                 saving_iterations,
                 ckpt_it,
                 ckpt_start,
                 debug_from,
                 expname
                 ):

        self.dataset = dataset
        self.hyperparams = hyperparams
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        self.checkpoint_iterations = ckpt_it
        self.checkpoint = ckpt_start
        self.expname = expname
        self.debug_from = debug_from

        self.tb_writer = prepare_output_and_logger(expname)
        self.gaussians = GaussianModel(dataset.sh_degree, hyperparams)
        dataset.model_path = args.model_path
        
        self.timer = Timer()
        self.scene = Scene(dataset, self.gaussians, load_coarse=None)
        self.timer.start()

        # Train the steps for the coarse field
        self.train_coarse()

        self.init_taining()

        try:
            self.W, self.H = self.scene.getTestCameras().dataset[0].image.shape[2], self.scene.getTestCameras().dataset[0].image.shape[1]
            self.fovy = self.scene.getTestCameras().dataset[0].FovY
        except:
            self.W, self.H = self.scene.getTestCameras()[0].image_width, self.scene.getTestCameras()[0].image_height
            self.fovy = self.scene.getTestCameras()[0].FoVy

        self.cam = OrbitCamera(self.W, self.H, r=30., fovy=self.fovy)
        self.mode = "rgb"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.time = 0.

        self.buffer = None
        self.coarse_stage = True


        self.ema_loss_for_log = 0.0
        self.ema_psnr_for_log = 0.0


        dpg.create_context()

        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=400,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.group():
                dpg.add_text("Training info: ")
                dpg.add_text("no data", tag="_log_iter")
                dpg.add_text("no data", tag="_log_loss")
                dpg.add_text("no data", tag="_log_psnr")
                dpg.add_text("no data", tag="_log_points")

            with dpg.collapsing_header(label="Testing info:", default_open=True):
                dpg.add_text("no data", tag="_log_psnr_test")
                dpg.add_text("no data", tag="_log_ssim")
                dpg.add_text("no data", tag="_log_ms-ssim")
                dpg.add_text("no data", tag="_log_d-ssim")
                dpg.add_text("no data", tag="_log_lpips")
                dpg.add_text("no data", tag="_log_lpipsa")
                

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                def callback_speed_control(sender):
                    self.time = dpg.get_value(sender)
                    self.need_update = True
                
                dpg.add_slider_float(
                    label="Time",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_speed_control,
                )
        
        
        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

        dpg.create_viewport(
            title="WavePlanes",
            width=self.W + 400,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()

    # gui mode
    def render(self):
        while dpg.is_dearpygui_running():
            if self.iteration <= self.final_iter:
                self.train_step()
                self.iteration += 1
            else: # Exit on last iteration
                exit()

            if (self.iteration % 100) == 0:
                self.test_step()

            self.viewer_step()
            dpg.render_dearpygui_frame()
    
    def train_step(self):

        # Start recording step duration
        self.iter_start.record()

        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

      
        # If data exists
        if self.opt.dataloader and not self.load_in_memory:
            try:
                viewpoint_cams = next(self.loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not self.random_loader:
                    viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    self.random_loader = True
                self.loader = iter(viewpoint_stack_loader)
        else:
            idx = 0
            viewpoint_cams = []
            
            # Construct batch for current step
            while idx < self.opt.batch_size :        
                viewpoint_cam = self.viewpoint_stack.pop(randint(0,len(self.viewpoint_stack)-1))
                
                # If viewpoint stack doesn't exist get it from the temp view point list
                if not self.viewpoint_stack :
                    self.viewpoint_stack =  self.temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1

            
            # If there are not cameras to load then end the current iteration
            if len(viewpoint_cams) == 0:
                return None
        
        # Render
        if (self.iteration - 1) == self.debug_from:
            self.pipe.debug = True

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        # Render and return preds
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage='fine',cam_type=self.scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if self.scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss = Ll1
        if self.hyperparams.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = self.gaussians.compute_regulation(self.hyperparams.time_smoothness_weight, self.hyperparams.l1_time_planes, self.hyperparams.plane_tv_weight)
            loss += tv_loss
        if self.opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += self.opt.lambda_dssim * (1.0-ssim_loss)

        # Backpass
        loss.backward()

        # Error if loss becomes nan
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        
        # Record end of step
        self.iter_end.record()

        # Log and save
        with torch.no_grad():
            self.timer.pause()
            if (self.iteration % 10) == 0:
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                self.ema_psnr_for_log = 0.4 * psnr_ + 0.6 * self.ema_psnr_for_log
                total_point = self.gaussians._xyz.shape[0]

                dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                dpg.set_value("_log_loss", f"Loss: {self.ema_loss_for_log} ")
                dpg.set_value("_log_psnr", f"PSNR: {psnr_}")
                dpg.set_value("_log_points", f"{total_point} total points")

                
            training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.checkpoint_iterations, self.scene, render, [self.pipe, self.background], 'fine', self.scene.dataset_type)
            
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, 'fine')
            
            # Render images
            if self.dataset.render_process:
                if (self.iteration < 1000 and self.iteration % 10 == 9) \
                    or (self.iteration < 3000 and self.iteration % 50 == 49) \
                        or (self.iteration < 60000 and self.iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(self.scene, self.gaussians, [self.test_cams[self.iteration%len(self.test_cams)]], render, self.pipe, self.background, "finetest", self.iteration,self.timer.get_elapsed_time(),self.scene.dataset_type)
                        render_training_image(self.scene, self.gaussians, [self.train_cams[self.iteration%len(self.train_cams)]], render, self.pipe, self.background, "finetrain", self.iteration,self.timer.get_elapsed_time(),self.scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
            
            self.timer.start()
            
            # Densification
            if self.iteration < self.opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
   
                opacity_threshold = self.opt.opacity_threshold_fine_init - self.iteration*(self.opt.opacity_threshold_fine_init - self.opt.opacity_threshold_fine_after)/(self.opt.densify_until_iter)  
                densify_threshold = self.opt.densify_grad_threshold_fine_init - self.iteration*(self.opt.densify_grad_threshold_fine_init - self.opt.densify_grad_threshold_after)/(self.opt.densify_until_iter)  
                
                if  self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify(densify_threshold, opacity_threshold, self.scene.cameras_extent, size_threshold, 5, 5, self.scene.model_path, self.iteration, "first")
                
                if  self.iteration > self.opt.pruning_from_iter and self.iteration % self.opt.pruning_interval == 0 and self.gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.prune(densify_threshold, opacity_threshold, self.scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if self.iteration % self.opt.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<360000 and self.opt.add_point:
                    self.gaussians.grow(5,5, self.scene.model_path, self.iteration, "fine")
                    # torch.cuda.empty_cache()
                
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)

            
            if (self.iteration in self.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
                torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_{'fine'}_" + str(self.iteration) + ".pth")

    @torch.no_grad()
    def viewer_step(self, specified_cam=None):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        custom_cam = MiniCam(
            self.cam.pose,
            self.W,
            self.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far, 
            time=self.time)
        
        buffer_image = render(custom_cam, self.gaussians, self.pipe, self.background, stage='fine', cam_type=self.scene.dataset_type)['render']

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        ender.record()
        self.need_update = True

        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        buffer_image = self.buffer_image
        dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!

    def train_step(self):

        # Start recording step duration
        self.iter_start.record()

        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

      
        # If data exists
        if self.opt.dataloader and not self.load_in_memory:
            try:
                viewpoint_cams = next(self.loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not self.random_loader:
                    viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    self.random_loader = True
                else:
                    viewpoint_stack_loader = self.viewpoint_stack_loader
                self.loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(self.loader)
        else:
            idx = 0
            viewpoint_cams = []
            
            # Construct batch for current step
            while idx < self.opt.batch_size :        
                viewpoint_cam = self.viewpoint_stack.pop(randint(0,len(self.viewpoint_stack)-1))
                
                # If viewpoint stack doesn't exist get it from the temp view point list
                if not self.viewpoint_stack :
                    self.viewpoint_stack =  self.temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1

            
            # If there are not cameras to load then end the current iteration
            if len(viewpoint_cams) == 0:
                return None
        
        # Render
        if (self.iteration - 1) == self.debug_from:
            self.pipe.debug = True

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        # Render and return preds
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage='fine',cam_type=self.scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if self.scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss = Ll1
        if self.hyperparams.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = self.gaussians.compute_regulation(self.hyperparams.time_smoothness_weight, self.hyperparams.l1_time_planes, self.hyperparams.plane_tv_weight)
            loss += tv_loss
        if self.opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += self.opt.lambda_dssim * (1.0-ssim_loss)

        # Backpass
        loss.backward()

        # Error if loss becomes nan
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        
        # Record end of step
        self.iter_end.record()

        # Log and save
        with torch.no_grad():
            self.timer.pause()
            if (self.iteration % 10) == 0:
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                self.ema_psnr_for_log = 0.4 * psnr_ + 0.6 * self.ema_psnr_for_log
                total_point = self.gaussians._xyz.shape[0]

                dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                dpg.set_value("_log_loss", f"Loss: {self.ema_loss_for_log} ")
                dpg.set_value("_log_psnr", f"PSNR: {psnr_}")
                dpg.set_value("_log_points", f"{total_point} total points")

                
            training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.checkpoint_iterations, self.scene, render, [self.pipe, self.background], 'fine', self.scene.dataset_type)
            
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, 'fine')
            
            # Render images
            if self.dataset.render_process:
                if (self.iteration < 1000 and self.iteration % 10 == 9) \
                    or (self.iteration < 3000 and self.iteration % 50 == 49) \
                        or (self.iteration < 60000 and self.iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(self.scene, self.gaussians, [self.test_cams[self.iteration%len(self.test_cams)]], render, self.pipe, self.background, "finetest", self.iteration,self.timer.get_elapsed_time(),self.scene.dataset_type)
                        render_training_image(self.scene, self.gaussians, [self.train_cams[self.iteration%len(self.train_cams)]], render, self.pipe, self.background, "finetrain", self.iteration,self.timer.get_elapsed_time(),self.scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
            
            self.timer.start()
            
            # Densification
            if self.iteration < self.opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
   
                opacity_threshold = self.opt.opacity_threshold_fine_init - self.iteration*(self.opt.opacity_threshold_fine_init - self.opt.opacity_threshold_fine_after)/(self.opt.densify_until_iter)  
                densify_threshold = self.opt.densify_grad_threshold_fine_init - self.iteration*(self.opt.densify_grad_threshold_fine_init - self.opt.densify_grad_threshold_after)/(self.opt.densify_until_iter)  
                
                if  self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify(densify_threshold, opacity_threshold, self.scene.cameras_extent, size_threshold, 5, 5, self.scene.model_path, self.iteration, "first")
                
                if  self.iteration > self.opt.pruning_from_iter and self.iteration % self.opt.pruning_interval == 0 and self.gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.prune(densify_threshold, opacity_threshold, self.scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if self.iteration % self.opt.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<360000 and self.opt.add_point:
                    self.gaussians.grow(5,5, self.scene.model_path, self.iteration, "fine")
                    # torch.cuda.empty_cache()
                
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)

            
            if (self.iteration in self.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
                torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_{'fine'}_" + str(self.iteration) + ".pth")

    @torch.no_grad()
    def test_step(self):
        idx = 0

        if self.iteration < (self.final_iter -1):
            idx = 0
            viewpoint_cams = []
            
            # Construct batch for current step
            while idx < 10:        
                viewpoint_cam = self.test_viewpoint_stack.pop(randint(0,len(self.test_viewpoint_stack)-1))
                
                # If viewpoint stack doesn't exist get it from the temp view point list
                if not self.test_viewpoint_stack :
                    self.test_viewpoint_stack =  self.test_temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
        else:
            viewpoint_cams = self.test_viewpoint_stack

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        # Render and return preds
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage='fine',cam_type=self.scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            images.append(image.unsqueeze(0))
            if self.scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        
        # Loss
        # breakpoint()
        ssims = []
        ms_ssims = []
        dssims = []
        psnrs = []
        lpipss = []
        lpipsa = []

        # Only compute extra metrics at the end of training -> can be slow
        if self.iteration < (self.final_iter -1):
            for idx in range(len(gt_images)):
                    psnrs.append(psnr(images[idx], gt_images[idx]))
                    ssims.append(ssim(images[idx], gt_images[idx]))            
            dpg.set_value("_log_psnr_test", "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            dpg.set_value("_log_ssim", "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        else:
            for idx in range(len(gt_images)):
                psnrs.append(psnr(images[idx], gt_images[idx]))

                ssims.append(ssim(images[idx], gt_images[idx]))            
                ms_ssims.append(ms_ssim(images[idx], gt_images[idx],data_range=1, size_average=True ))
                dssims.append((1-ms_ssims[-1])/2)


                lpipss.append(lpips(images[idx], gt_images[idx], net_type='vgg'))
                lpipsa.append(lpips(images[idx], gt_images[idx], net_type='alex'))
            
            
            dpg.set_value("_log_psnr_test", "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            dpg.set_value("_log_ssim", "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            dpg.set_value("_log_ms-ssim", "MS-SSIM : {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
            dpg.set_value("_log_d-ssim", "D-SSIM : {:>12.7f}".format(torch.tensor(dssims).mean(), ".5"))
            dpg.set_value("_log_lpips", "LPIPS-VGG : {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            dpg.set_value("_log_lpipsa", "LPIPS-Alex : {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))       




def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument('--gui', action='store_true', default=False)

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.gui:
        gui = GUI(args=args, hyperparams=hp.extract(args), dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,
                ckpt_it=args.checkpoint_iterations, ckpt_start=args.start_checkpoint, debug_from=args.debug_from, expname=args.expname)

        gui.render()

    # training( args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")