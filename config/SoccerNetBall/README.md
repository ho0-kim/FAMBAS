<div align="left">

# Setting up the Configurations

Here we describe the different parameters set in each configuration file:

- _frame_dir:_ Directory where frames are stored.
- _save_dir:_ Directory to save model checkpoints, predictions, processed datasets...
- _store_mode:_ `store` if it's the first time running the script to prepare and store dataset information, or `load` to load previously stored information.
- _batch_size:_ Batch size.
- _clip_len:_ Length of the clips in number of frames.
- _crop_dim:_ Integer value specifying the resolution (in pixels) to crop the frame into a square. Use `-1` to indicate no cropping.
- _dataset:_ Name of the dataset (`soccernetball`).
- _event_team:_ Boolean indicating if detecting also team side in addition to spotting.
- _radi_displacement:_ Radius of displacement used.
- _epoch_num_frames:_ Number of frames used per epoch.
- _feature_arch:_ Feature extractor architecture (`rny002_gsf` or `rny008_gsf`).
- _learning_rate:_ Learning rate.
- _mixup:_ Boolean indicating whether to use mixup or not.
- _modality:_ Input modality used (`rgb`).
- _num_classes:_ Number of classes for the current dataset.
- _num_epochs:_ Number of epochs for training.
- _warm_up_epochs:_ Number of warm-up epochs.
- _start_val_epoch:_ Epoch where validation evaluation starts.
- _temporal_arch:_ Temporal architecture used (`ed_sgp_mixer`).
- _n_layers:_ Number of blocks/layers used for the temporal architecture.
- _sgp_ks:_ Kernel size of the SGP and SGP-Mixer layers.
- _sgp_r:_ $r$ factor in SGP and SGP-Mixer layers.
- _only_test:_ Boolean indicating if only inference is performed or training + inference.
- _criterion:_ Criterion used for validation evaluation (`map`, `loss`).
- _num_workers:_ Number of workers.
- _joint_train:_ Additional dataset information for joint training (SoccerNet Action Spotting -soccernet- in our case).
    - _frame_dir:_ Directory where frames are stored (for additional dataset - snas).
    - _dataset:_ Additional dataset used (`soccernet`).
    - _num_classes:_ Nº of classes of the additional dataset.
