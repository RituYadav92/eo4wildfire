
from pathlib import Path
from easydict import EasyDict as edict
import os

project_dir = Path(os.getcwd())
cfg = edict(
        project_dir = str(project_dir),
        data_folder = str(project_dir / "Data"),
        # rootPath = str(workspace / "Historical_Wildfire_Dataset"),
        
        # # tif
        # input_folder = "A0_SAR_Input_Data",
        # transfer_folder = "A0_Progression_Data"

        # PNG
        input_folder = "A0_SAR_Input_PNG", 
        transfer_folder = "A0_Progression_PNG",

        # sampling config
        patchsize = 256,
        num_patch_per_image = 10,
        train_val_split_rate = 0.7,
        random_state = 42,

        # model config
        ARCH = 'UNet',
        ENCODER = 'resnet50', # 'mobilenet_v2'
        learning_rate = 1e-5,
        weight_decay = 1e-4,
        BATCH_SIZE = 16,

        max_score = 0,
        max_epoch = 20,
        size_of_train = 3072,

        # loss
        gamma = 1, # dice
        alpha = 1, # focal
        beta = 1e-4, # tv

        ref_mode = 'SAR', # 'SAR'
        
        ENCODER_WEIGHTS = 'imagenet',
        ACTIVATION = 'sigmoid', # could be None for logits or 'softmax2d' for multicalss segmentation

        CLASSES = ['fire'],
        DEVICE = 'cuda',
        verbose = True,
    )