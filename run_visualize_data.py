import torch
from easydict import EasyDict as edict
from datasets.data_samping import DataSampler
from pathlib import Path
from prettyprinter import pprint
from imageio import imread, imsave
import json
from datasets.data_provider import Dataset
import matplotlib.pyplot as plt


# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')

# if __name__ == '__main__':
#     run()

from config.configuration import cfg

""" Data Sampling """
data_sampler = DataSampler(cfg)
# data_sampler()
# cfg.data_sampler = data_sampler

""" Change Configuration """
cfg.gamma = 1
cfg.alpha = 1
cfg.beta = 1e-4
# for ref_mode in ['SAR']:
cfg.ref_mode = 'SAR'

pprint(cfg)

""" Model Training """
from models.seg_model import SegModel
cfg.data_sampler = data_sampler

train_dataset = Dataset(
            str(data_sampler.train.patchDir), #self.x_train_dir, 
            str(data_sampler.train.maskDir_SAR), #self.y_train_dir, 
            # augmentation=get_training_augmentation(), 
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=cfg.CLASSES,
        )


# rand_list = np.random.sample(range(0, num), 10)
for i in range(100, 10):
    image, mask = train_dataset[i] # get some sample
    visualize(
        saveFlag=False,
        image=image, 
        cars_mask=mask.squeeze(),
    )






