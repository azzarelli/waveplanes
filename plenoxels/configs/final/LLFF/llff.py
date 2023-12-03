config = {
    "expname": "my scene name",
    "logdir": "./logs/LLFF",
    "device": "cuda:0",
    'wandbproject':'KW LLFF',

    # Wavelet Configs
    'wave_level': 2,
    'use_wavelet_psn':True, # If False use K-Planes
    'use_wavelet_field':True, # If false use K-Planes
    'fusion' : 'MUL', # Options: MUL, ADD, ZMM, ZAM
    'static_scene':True,
    
    # Model settings
    "density_activation": "trunc_exp",
    "concat_features_across_scales": True,
    "linear_decoder": True,
    "linear_decoder_layers": 4, 

    # Data settings
    "data_downsample": 4,
    "data_dirs": ["data/LLFF/fern"],
    
    # Data settings for LLFF
    "hold_every": 8,
    "contract": False,
    "ndc": True,
    "near_scaling": 0.89,
    "ndc_far": 2.6,

    # Optimization settings
    "num_steps": 40_001,
    "batch_size": 4096,
    "eval_batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 0.01,

    # Regularization
    "plane_tv_weight": 1e-4,
    "plane_tv_weight_proposal_net": 1e-4,
    "l1_proposal_net_weight": 0,
    "histogram_loss_weight": 1.0, 
    "distortion_loss_weight": 0.001,

    # Training settings
    "train_fp16": True,
    "save_every": 40000,
    "valid_every": 40000,
    "save_outputs": True,

    # Raymarching settings
    "num_samples": 48,
    "single_jitter": False,
    # proposal sampling
    "num_proposal_samples": [256, 128],
    "use_same_proposal_network": False,
    "use_proposal_weight_anneal": True,
    "proposal_net_args_list": [
     {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 1], 'wave_level':2},
     {'num_input_coords': 4, 'num_output_coords': 16, 'resolution': [256, 256, 256, 1], 'wave_level':2}
    ],


    "grid_config": [
        { # Main Field
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 16,
        "grid_dimensions": 2,

        'feature_size': 64,
        'resolution': [512, 512, 512, 1],

        'wave': 'coif4', # bior4.4, haar
        'wave_mode': 'periodization', # periodization
        'regularise_wavelet_coeff':False, # directly regularise wavelet coefficients
        'show_planes_before_run':False,
        },
        { # Proposal Net (i.e Density Field)
        'wave': 'coif4',
        'wave_mode':'periodization',
        'regularise_wavelet_coeff':False,
        'concat_features_across_scales':True
        }
    ],
}
