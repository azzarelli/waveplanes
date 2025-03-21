config = {
 'expname': 'hell',
 'logdir': './logs/hellwarrior',
 'device': 'cuda:0',
  'wandbproject':'W3DNF',


 'use_wavelet_psn':True,
 'use_wavelet_field':True,
 'fusion':'MUL', # MUL, ADD, ZMM, ZAM


 'data_downsample': 2.0,
 'data_dirs': ['/home/barry/data/dnerf/hellwarrior'],
 'contract': False,
 'ndc': False,
 'isg': False,
 'isg_step': -1,
 'ist_step': -1,
 'keyframes': False,
 'scene_bbox': [[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]],

 # Optimization settings
 'num_steps': 60005,
 'batch_size': 4096,
 'scheduler_type': 'warmup_cosine',
 'optim_type': 'adam',
 'lr': 0.01,

 # Regularization
 'distortion_loss_weight': 0.00,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.00001,
 'l1_time_planes_proposal_net': 0.00001,
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'time_smoothness_weight': 0.1,
 'time_smoothness_weight_proposal_net': 0.001,

 # Training settings
 'valid_every': 10000,
 'save_every': 30000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False, 
 'num_samples': 48,
 'num_proposal_samples': [256, 128],

 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
     {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [32, 32, 32, 50], 'wave_level':2},
     {'num_input_coords': 4, 'num_output_coords': 16, 'resolution': [64, 64, 64, 100], 'wave_level':2}

 ],

 # Model settings
 'concat_features_across_scales': True,
 'wave_level': 3,
 'density_activation': 'trunc_exp',
 'linear_decoder': True,
 'linear_decoder_layers': 4,

 'grid_config': [
     # For Main Fields
     {
    'grid_dimensions': 2,
    'input_coordinate_dim': 4,
    'output_coordinate_dim': 32,
    'resolution': [128, 128, 128, 100],
    'feature_size': 64,
    'wave': 'coif4', # bior4.4, haar
    'wave_mode': 'periodization', # periodization
    'regularise_wavelet_coeff':False,
    'show_planes_before_run':False
    },
    # For Proposal Net (i.e Density Field)
    {
    'wave': 'coif4',
    'wave_mode':'periodization',
    'regularise_wavelet_coeff':False,
    'concat_features_across_scales':True
    }]

}