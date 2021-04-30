## Configs

This directory contains the serialized configs used for training, evaluation, and inference.
The original configs can be found in `rvfn/config/`.

The configurations are broken up into three parts:

- `pipeline`: Contains the configurations that specify the entire 
  inference pipeline behavior. This includes dataset preprocessing configs, 
  model configs, target configs, and inference configs.
- `train`: Configurations and hyper-parameters for the training process.
- `eval`: Configurations for the evaluation process.

Multiple variations of the pipeline configs are provided in the `pipeline/` directory.
These variations can be automatically generated from the original python configs in 
`rvfn/config/` using
`rvfn/tools/make_configs.py`.

Below is a summary of available configs and what they do. For fruther detail see the 
corresponding files in `rfvn/config/`.
### Pipeline
#### Dataset
```python
"root_path": "/nuscenes",        # Path to dataset root
"version": "v1.0-trainval",      # Dataset version
"cam_name": "CAM_FRONT",         # The camera to use for FOV limiting and fusion
"radar_name": "RADAR_FRONT",     # The radar sensor to use
"lidar_sweeps": 3,
"use_lidar": true,               # At least one of use_lidar or use_radar must be true
"use_radar": false,
"use_rgb": false,                # Fuse RGB features from camera. Will be ignored if use_lidar is false
"radar_sweeps": 3,
"min_dist": 1.0,                 # Removes points closer than this in meters
"fill_type": null,               # One of {null, 'ipbasic', 'knn', 'maskconv'}
"img_size": [                    # Dataset image size in pixels
  1600, 900
],
"pointcloud_range": [            # In x, y, and z direction in meters
  [1.0, 50],                     # Will remove points and bboxes that are outside this boundary
  [-20.0, 20.0],                 # Can also be null to indicate no point removal
  [-1.0, 3.0]
],
"auto_offset": true,             # If True will offset point and bbox coordinates such that all
                                 #  coordinates become non-negative
"coord_offsets": [               # How much to offset the coordinates of points and bboxes in each
  0, 20.0, 1                     #  axis. Will be ignored if 'auto_offset' is true
],
"max_points": 40000,             # Maximum number of lidar points in the pointcloud
"categories": [                  # A list of lists, where each sublist indicates categories
  ["car"]                        #  that will be mapped to the same label.
],
        
"voxel_grid_config": {
  "voxel_size": [                # In x, y, and z direction in meters.
    0.2, 0.2, 0.4
  ],
  "max_points_per_voxel": 40
},
        
"augmentation_config": {
  "box_rotation": 0.314,         # Rotation limit for boxes and points inside them in radians.
                                 # Rotation will be uniformly between -box_rotation and
                                 #  +box_rotation
  "box_translation": [           # Translation standard deviation in x, y, and z axes
    1.0, 1.0, 0.3
  ],
  "global_rotation": 0.314,      # Rotation limit for the point cloud
                                 # Rotation will be uniformly between -global_rotation and
                                 #  +global_rotation
  "global_scale": 0.05           # Scaling limit for the point cloud
                                 # Scaling will be uniformly between 1 - global_scale and
}                                #  1 + global_scale

```

#### Model
```python
"svfeb_config": {               # Parameters for the Stacked Voxel Feature Encoder
    "in_channels": 7,           # Each voxel has 3 global and 3 local coordinates for a total of 6
                                # Other features such as lidar intensity or radar RCS must be
                                #  added to this number to yield the total in_channels
    "hidden_channels": 32,
    "out_channels": 128
},
"cmb_config": {                 # Parameters for the Convolutional Middle Block
    "input_spatial_size": [     # Must be set accurately. See rvfn/config/model.py for details
        11, 204, 252
    ],
    "expected_output_spatial_size": [
        3, 200, 248
    ],
    "in_channels": 128,
    "out_channels": 64
},
"cob_config": {                 # Parameters for the fully convolutional output block
    "in_channels": 192,         # See rvfn/config/model.py for details
    "hidden_channels": 256,
    "in_spatial_size": [        # Feature map shape will be half of this
        200, 248
    ]
},
"head_config": {                # Parameters for the prediction heads
    "num_classes": 2,           # Detection classes ('car' and 'background' in this case)
    "anchors_per_position": 2,  # Number of anchors per feature map position
    "in_channels": 768
},
"middle_block": "sparse"        # One of {'sparse', 'normal'}
```

#### Target
```python
"pos_threshold": 0.35,
"neg_threshold": 0.3,
"pos_dist_threshold": 0.5,
"anchor_config": {
    "feature_map_shape": [      # Feature map shape of the COB.
        100, 124
    ],
    "anchor_sizes": [           # Size and rotation of produced anchors.
        [1.92, 4.62, 1.69]      # Anchor generator will create
    ],                          #  len(anchor_sizes) * len(anchor_rotations) anchors per
    "anchor_rotations": [       #  position in the feature map.
        0, 1.5707963267948966
    ],
    "anchor_range": [           # The ranges in which the anchors will be spread in meters
        [0, 50],                # x
        [0, 40]                 # y
    ],
    "center_z": 2.0             # The center of anchors in z axis
},
"sine_yaw_targets": true        # Use sine of yaw difference as regression targets
```

#### Infer
```python
"nms_threshold": 0.1,           # IoU threshold for non-maximum-supression
"min_confidence": 0.001         # Filter out prediction with scores of less than this
```

### Train
```python
"loss_config": {
    "score_loss_config": {          # Loss function for the classification head
        "criteria": "binary_focal", # Type of loss
        "criteria_config": {
            "gamma": 0.0,
            "alpha": 0.6,
            "reduce": false,        # Sum over anchors and get the mean across minibatches
            "logits": false         # whether the inputs are logits
        },
        "normalize_by_type": true   # Divide the loss for positive and negative anchors
                                    #  the number of each
    },
    "reg_loss_config": {            # Loss for the regression head
        "criteria": "smoothL1"
    },
    "dir_loss_config": {            # Loss for the direction head
        "criteria": "binary_cross_entropy"
    },
    "score_loss_weight": 2.0,       # Weight of each part in the total loss
    "reg_loss_weight": 1.0,
    "dir_loss_weight": 0.2
},
"augment": true,                    # Do on-the-fly augmentations of the training set
"batch_size": 4,
"optimizer": "AMSGrad",
"learning_rate": 0.001,
"epsilon": 1e-08,                   # Epsilon value used for the optimizer
"weight_decay": 0.0,
"epochs": 1000,
"out_path": "pers/out/",            # Where to write checkpoints, logs, and configs
"log_interval": 500,                # Print a log every this many bathces
"checkpoint_interval": 4,           # Save a checkpoint after this many epochs
"eval_interval": 4,                 # Evaluate on the train and validation sets after
                                    #  this many epochs
"eval_on_train_fraction": 0.1,      # Fraction of training set to evaluate on
                                    #  (this is separate from the evaluation on the
                                    #   validation set)
"data_loader_workers": 6,
"device_ids": [                     # ID of GPU devices to use
    0
]
```

### Eval
```python
"distance_thresholds": [           # Distance thresholds for matching
    0.5, 1, 2, 4                   #  See nuScenes evaluation for more detail
],
"data_loader_workers": 6,
"batch_size": 4,
"device_ids": [                    # ID of GPU devices to use
    0
]
```
