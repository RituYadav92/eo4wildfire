import torch
from easydict import EasyDict as edict
from datasets.data_samping import DataSampler
from pathlib import Path
from prettyprinter import pprint
from imageio import imread, imsave
import json


# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')

# if __name__ == '__main__':
#     run()

from config.configuration import cfg

""" Data Sampling """
data_sampler = DataSampler(cfg)
data_sampler()
# cfg.data_sampler = data_sampler

""" Change Configuration """
cfg.alpha = 0.5 # focal loss
cfg.ref_mode = 'SAR'
for ARCH in ['UNet']:
    cfg.ARCH = ARCH
    for beta in [5e-4]: #  
        cfg.beta = beta
    
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

        """ Evaluation """
        from evaluation.evaluator_11_wildfires import Evaluator
        evaluator = Evaluator(cfg)
        evaluator.model_testing()
        evaluator.model_transfer()


