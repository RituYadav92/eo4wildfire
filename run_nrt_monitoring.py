import torch
from easydict import EasyDict as edict
from datasets.data_samping import DataSampler, NRT_DataSampler
from pathlib import Path
from prettyprinter import pprint
from imageio import imread, imsave
import os, json


project_dir = Path(os.getcwd())
cfg = edict(
    project_dir = str(project_dir),
    # data_folder = str(project_dir / "BC_Wildfire_Data"),
    nrt_data_folder = str(project_dir / "Data/Temporal_Progressions_Data/FraserIsland_Progression_Data_20m"),

    # PNG
    input_folder = "A0_Progression_PNG", 
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

    max_score = 0, # IoU
    max_epoch = 10,
    size_of_train = 3072,

    # loss
    gamma = 1, # dice
    alpha = 0.5, # focal
    beta = 2e-5, # tv

    ref_mode = 'SAR', # 'SAR', 'Opt', 'optSAR'
    
    ENCODER_WEIGHTS = 'imagenet',
    ACTIVATION = 'sigmoid', # could be None for logits or 'softmax2d' for multicalss segmentation

    CLASSES = ['fire'],
    DEVICE = 'cuda',
    verbose = True,
)

""" Data Sampling """
# data_sampler = DataSampler(cfg)
data_sampler = NRT_DataSampler(cfg)
print("data_sampler")
data_sampler()
cfg.data_sampler = data_sampler

for ref_mode in ['SAR']:
    cfg.ref_mode = ref_mode

    pprint(cfg)

    """ Model Training """
    from models.seg_model_nrt import SegModel
    cfg.data_sampler = data_sampler

    seg_model = SegModel(cfg)
    cfg.modelPath = str(seg_model.savePath)

    """ Save Configuration before run """
    cfg_file = Path(cfg.modelPath) / 'config.json'
    with open(str(cfg_file), 'w') as fp:
        if 'data_sampler' in cfg.keys(): del cfg['data_sampler']
        json.dump(cfg, fp)

    seg_model.run_nrt_experiment()

    # specify the data_folder for model transfering
    # cfg.data_folder = Path(os.getcwd()) / "Data"
    # """ Evaluation """
    # from evaluation.BC_evaluator import Evaluator
    # evaluator = Evaluator(cfg)
    # evaluator.model_testing()
    # evaluator.model_transfer()

