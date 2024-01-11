import torch
from easydict import EasyDict as edict
from datasets.data_samping import DataSampler
from pathlib import Path
from prettyprinter import pprint
from imageio import imread, imsave
import os, json


project_dir = Path(os.getcwd())
cfg = edict(
    project_dir = str(project_dir),
    data_folder = str(project_dir / "Data" / "Historical_Wildfire_Dataset" / "BC_SAR4Wildfire_Dataset"),

    # PNG
    input_folder = "A0_SAR_Input_PNG", 
    transfer_folder = "A0_Progression_PNG",

    # sampling config
    patchsize = 256,
    num_patch_per_image = 500, #500
    train_val_split_rate = 0.7,
    random_state = 42,

    # model config
    ARCH = 'UNet',
    ENCODER = 'resnet50', # 'mobilenet_v2'
    learning_rate = 1e-5,
    weight_decay = 1e-4,
    BATCH_SIZE = 32,

    max_score = 0.1, # If IoU > max_score, start to save model
    max_epoch = 2, # max iteration
    size_of_train = 3072, # size of data for training

    # loss
    gamma = 1, # dice
    alpha = 0.5, # focal
    beta = 2e-5, # tv

    ref_mode = 'SARREF', # 'SARREF', 'optSAR', 'OptREF'
    
    ENCODER_WEIGHTS = 'imagenet',
    ACTIVATION = 'sigmoid', # could be None for logits or 'softmax2d' for multicalss segmentation

    CLASSES = ['fire'],
    DEVICE = 'cuda',
    verbose = True,
)

""" Data Sampling """
data_sampler = DataSampler(cfg)
print("data_sampler")
data_sampler()

for ref_mode in ['SARREF']: #'SARREF', 'OptSAR', 'OptREF'
    cfg.ref_mode = ref_mode
    pprint(cfg)

    """ Model Training """
    from models.seg_model import SegModel
    cfg.data_sampler = data_sampler

    seg_model = SegModel(cfg)
    cfg.modelPath = str(seg_model.savePath)

    """ Save Configuration before run """
    cfg_file = Path(cfg.modelPath) / 'config.json'
    with open(str(cfg_file), 'w') as fp:
        if 'data_sampler' in cfg.keys(): del cfg['data_sampler']
        json.dump(cfg, fp)

    seg_model.run_experiment()

    # specify the data_folder for model transfering
    
    """ Evaluation """
    from evaluation.BC_evaluator import Evaluator
    cfg.data_folder = str(Path(os.getcwd()) / "BC_Wildfire_Data")
    evaluator = Evaluator(cfg)
    # evaluator.model_testing()
    # evaluator.model_transfer()

    cfg.data_folder = str(Path(os.getcwd()) / "Data")
    evaluator = Evaluator(cfg)
    evaluator.model_transfer()


