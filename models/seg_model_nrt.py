
import kornia
import torch
import torch.nn.functional as F
import numpy as np
# import segmentation_models_pytorch.segmentation_models_pytorch as smp
import segmentation_models_pytorch as smp
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu

from easydict import EasyDict as edict
import os, sys
from pathlib import Path
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from astropy.visualization import PercentileInterval
interval_95 = PercentileInterval(95.0)

from imageio import imread, imsave
import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
import pandas as pd
from prettyprinter import pprint
# from datasets.data_provider import Dataset
from preprocessing.augment import get_training_augmentation, get_validation_augmentation, get_preprocessing
from datasets.data_visualize import visualize
from evaluation.evaluator import Evaluator

# class import
f_score = smp.utils.functional.f_score

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
diceLoss = smp.utils.losses.DiceLoss(eps=1)

tv_loss = kornia.losses.TotalVariation()
tv_loss.__name__ = 'tv_loss'

mseLoss = torch.nn.MSELoss(reduction='mean')
mseLoss.__name__ = 'mse_loss'

AverageValueMeter =  smp.utils.train.AverageValueMeter

from models.focal_loss import FocalLoss
focal_loss = FocalLoss()

metrics = [
    diceLoss,  
    focal_loss, 
    tv_loss,
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore()
]

def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    # CLASSES = ['fire', 'sky', 'building', 'pole', 'road', 'pavement', 
    #            'tree', 'signsymbol', 'fence', 'car', 
    #            'pedestrian', 'bicyclist', 'unlabelled']

    CLASSES = ['nonfire', 'fire']
    
    def __init__(
            self, 
            dataName,
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,

    ):
        # use all data in train datapath
        # self.ids = os.listdir(images_dir)

        ## All data Before this date.
        self.ids = [img_id for img_id in os.listdir(images_dir) if img_id.split("_")[0] <= dataName.split("_")[0]]
        print(f"size of trainSet: {len(self.ids)}")
        # pprint(self.ids)
        ## Only this date
        # self.ids = [img_id for img_id in os.listdir(images_dir) if img_id.split("_")[0] == dataName.split("_")[0]]

        # self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(self.masks_fps[i], 0) / 255.0) #  / 255
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class SegModel:
    def __init__(self, cfg):
        self.time_now = (datetime.now() + timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
        self.cfg = cfg

        self.expmPath = Path(cfg.project_dir) / 'Experiments'
        # self.expmPath = Path(cfg.nrt_data_folder) / 'Experiments'
        if not os.path.exists(self.expmPath): os.mkdir(self.expmPath)
        self.firename = os.path.split(self.cfg.nrt_data_folder)[-1].split("_")[0]
        self.savePath = self.expmPath / f"{self.firename}_{self.time_now}_{cfg.ref_mode}_{cfg.ARCH}_{cfg.ENCODER}_a_{cfg.alpha}_b_{cfg.beta}_trainSize_{cfg.size_of_train}"
        self.model_url = str(self.savePath / 'best_model.pth')

        if not os.path.exists(self.expmPath): os.mkdir(self.expmPath)
        if not os.path.exists(self.savePath): os.mkdir(self.savePath)
        print(f"train expmPath: {self.expmPath}")
        print(f"train savePath: {self.savePath}")

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.cfg.ENCODER, self.cfg.ENCODER_WEIGHTS)
        self.model = self.choose_network_architecture()

        self.ref_mode = cfg.ref_mode
        self.data_sampler = cfg.data_sampler
        self.set_train_val_dir()


    def set_train_val_dir(self):
        # set train and valid dir
        self.x_train_dir = str(self.data_sampler.train.patchDir)
        self.x_valid_dir = str(self.data_sampler.val.patchDir)

        if self.ref_mode == 'SAR':
            self.y_train_dir = str(self.data_sampler.train.maskDir_SAR)
            self.y_valid_dir = str(self.data_sampler.val.maskDir_SAR)
        elif self.ref_mode == 'OptSAR':
            self.y_train_dir = str(self.data_sampler.train.maskDir_optSAR)
            self.y_valid_dir = str(self.data_sampler.val.maskDir_optSAR)
        elif self.ref_mode == 'Opt':
            self.y_train_dir = str(self.data_sampler.train.maskDir_Opt)
            self.y_valid_dir = str(self.data_sampler.val.maskDir_Opt)
        else:
            pass
        self.x_test_dir = self.x_valid_dir
        self.y_test_dir = self.y_valid_dir
        

    def train_val_loader(self, num=-1):
        train_dataset = Dataset(
            dataName=self.dataName,
            images_dir=self.x_train_dir, 
            masks_dir=self.y_train_dir, 
            augmentation=get_training_augmentation(), 
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.cfg.CLASSES,
        )

        valid_dataset = Dataset(
            dataName=self.dataName,
            images_dir=self.x_valid_dir, 
            masks_dir=self.y_valid_dir, 
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.cfg.CLASSES,
        )

        if num > 0:
            print(f"Size of Train/Val: {num}")
            train_dataset = Subset(train_dataset, np.random.choice(len(train_dataset), num))
            # valid_dataset = Subset(valid_dataset, np.random.choice(len(valid_dataset), int(num/4))

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=False, num_workers=8)

        dataloaders = {'Train': train_loader, 'Val': valid_loader}

        return dataloaders

    
    def run_nrt_experiment(self):

        self.history_logs = edict()
        self.history_logs['Train'] = []
        self.history_logs['Val'] = []

        for dataName in sorted(os.listdir(Path(self.cfg.nrt_data_folder) / self.cfg.input_folder)):
            self.dataName = dataName
            print(f"\n==> {self.dataName}")

            self.dataloaders = self.train_val_loader(num=self.cfg.size_of_train)
    
            self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)])
            # self.history_logs = {'Train': np.zeros((len(metrics)+1, self.cfg.max_epoch)), 
                            # 'Val': np.zeros((len(metrics)+1, self.cfg.max_epoch))}

            # --------------------------------- Train -------------------------------------------
            for epoch in range(0, self.cfg.max_epoch):
                print(f"\n==> train epoch: {epoch}/{self.cfg.max_epoch}")
                valid_logs = self.train_one_epoch(epoch)
                    
                # do something (save model, change lr, etc.)
                if self.cfg.max_score < valid_logs['iou_score']:
                    max_score = valid_logs['iou_score']
                    torch.save(self.model, self.model_url)
                    print('Model saved!')
                    
                if epoch == 10:
                    self.optimizer.param_groups[0]['lr'] = self.cfg.learning_rate * 0.1
                    print(f"Decrease decoder learning rate to {self.optimizer.param_groups[0]['lr']}!")

                # save learning history
                self.plot_and_save_learnHistory()

            self.cfg.data_folder = self.cfg.nrt_data_folder
            self.cfg.modelPath = self.savePath
            self.evaluator = Evaluator(self.cfg)

            url = Path(self.cfg.nrt_data_folder) / self.cfg.input_folder / self.dataName
            print(url)
            predMap = self.evaluator.inference(url, self.savePath)
                   
         
    
    def train_one_epoch(self, epoch):
        self.model.to(self.cfg.DEVICE)

        for phase in ['Train', 'Val']:
            if phase == 'Train':
                self.model.train()
            else:
                self.model.eval()

            self.step(phase)            
            # self.history_logs[phase][:, epoch] = [self.logs["total_loss"]] + [self.logs[metrics[i].__name__] for i in range(0, len(metrics))]
            temp = [self.logs["total_loss"]] + [self.logs[metrics[i].__name__] for i in range(0, len(metrics))]
            self.history_logs[phase].append(temp)

            if phase == 'Val':
                valid_logs = self.logs
                return valid_logs

    def step(self, phase):
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}

        with tqdm(iter(self.dataloaders[phase]), desc=phase, file=sys.stdout, disable=not self.cfg.verbose) as iterator:
            for sample in iterator:
                x, y = sample[0].to(self.cfg.DEVICE), sample[1].to(self.cfg.DEVICE)
                self.optimizer.zero_grad()

                if self.cfg.ARCH == 'FCN':
                    y_pred = self.model.forward(x)['out']
                else: 
                    y_pred = self.model.forward(x)

                dice_loss_ =  self.cfg.gamma * diceLoss(y_pred, y)
                focal_loss_ = self.cfg.alpha * focal_loss(y_pred, y)
                tv_loss_ = self.cfg.beta * torch.mean(tv_loss(y_pred))

                loss_ = dice_loss_ + focal_loss_ + tv_loss_

                # update loss logs
                loss_value = loss_.cpu().detach().numpy()
                loss_meter.add(loss_value)
                # loss_logs = {criterion.__name__: loss_meter.mean}
                loss_logs = {'total_loss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in metrics:
                    if 'tv' in metric_fn.__name__:
                        metric_value =  self.cfg.beta* torch.mean(metric_fn(y_pred)).cpu().detach().numpy()
                    elif 'focal' in metric_fn.__name__:
                        metric_value = self.cfg.alpha * metric_fn(y_pred, y).cpu().detach().numpy()
                    elif 'dice' in metric_fn.__name__:
                        metric_value = self.cfg.gamma * metric_fn(y_pred, y).cpu().detach().numpy()
                    else:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                # print(logs)

                if self.cfg.verbose:
                    s = format_logs(logs)
                    iterator.set_postfix_str(s)

                if phase == 'Train':
                    loss_.backward()
                    self.optimizer.step()

        self.logs = logs
        

    def choose_network_architecture(self):
        # UNet
        if self.cfg.ARCH == 'UNet':
            print(f"===> Network Architecture: {self.cfg.ARCH}")
            # create segmentation model with pretrained encoder
            Unet = smp.Unet(
                encoder_name = self.cfg.ENCODER, 
                encoder_weights = self.cfg.ENCODER_WEIGHTS, 
                classes = len(self.cfg.CLASSES), 
                activation = self.cfg.ACTIVATION,
            )
            return Unet

        # UNet
        if self.cfg.ARCH == 'DeepLabV3+':
            print(f"===> Network Architecture: {self.cfg.ARCH}")
            # create segmentation model with pretrained encoder
            Unet = smp.DeepLabV3Plus(
                encoder_name = self.cfg.ENCODER, 
                encoder_weights = self.cfg.ENCODER_WEIGHTS, 
                classes = len(self.cfg.CLASSES), 
                activation = self.cfg.ACTIVATION,
                # in_channels = 1
            )
            return Unet
        
        # FCN
        if self.cfg.ARCH == 'FCN':
            print(f"===> Network Architecture: {self.cfg.ARCH}")
            if self.cfg.ENCODER == 'resnet50':
                fcn = models.segmentation.fcn_resnet50(pretrained=True)
            if self.cfg.ENCODER == 'resnet101':
                fcn = models.segmentation.fcn_resnet101(pretrained=True)
                
            # FCNHead = models.segmentation.fcn.FCNHead
            UnetSegHead =  smp.base.heads.SegmentationHead
            fcn.classifier[4] = UnetSegHead(512, 1, kernel_size=1, activation='sigmoid') # FCNHead(2048, 1)
            return fcn

    
    def plot_and_save_learnHistory(self):
        train_history = np.array(self.history_logs['Train']).transpose()
        val_history = np.array(self.history_logs['Val']).transpose()
        trainValHis = np.concatenate((train_history, val_history), axis=0)
        itemList = ['total_loss'] + [metrics[i].__name__ for i in range(0, len(metrics))]
        columes_names = [f"{phase}_{item}" for phase in ['Train', 'Val'] for item in itemList]

        df = pd.DataFrame(trainValHis, columns=range(0, trainValHis.shape[1]), index = columes_names)

        for item in itemList:
            plt.figure()
            df.transpose()[[f'Train_{item}', f'Val_{item}']].plot()
            if 'loss' not in item:
                plt.ylim([0, 1])
            
            if 'dice_loss' in item: # added Nov.16, 2020
                plt.ylim([0, 1])

            plt.tight_layout()
            plt.savefig(self.savePath / f"learn_curve_{item}.png")
            plt.close('all')

        df.to_csv(str(self.savePath / "train_val_History.csv"))



# optimizer = torch.optim.Adam([ 
#     dict(params=model.parameters(), lr=0.0001),
# ])

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()

#     from datasets.data_samping import DataSampler
#     sample_cfg = edict(
#             rootPath = Path("E:\PyProjects\PyTorch_TV_Transfer_Learning\data"),

#             patchsize = 256,
#             num_patch_per_image = 10,
#             train_val_split_rate = 0.7,
#             random_state = 42,
#         )

#     data_sampler = DataSampler(sample_cfg)
#     data_sampler()
#     pprint(dataSampler.cfg)

#     model_cfg = edict(
#         ARCH = 'UNet',
#         ENCODER = 'resnet18', # 'mobilenet_v2'
#         ENCODER_WEIGHTS = 'imagenet',
#         ACTIVATION = 'sigmoid', # could be None for logits or 'softmax2d' for multicalss segmentation
#         alpha = 1,
#         beta = 1e-4,

#         CLASSES = ['fire'],
        
#         DEVICE = 'cuda',
        
#         verbose = True,
#         max_score = 0.2,
#         max_epoch = 20,
#         size_of_train = 2048,
        
#         ref_mode = 'SAR',
#         data_sampler = data_sampler
#     )

#     seg_model = SegModel(model_cfg)
#     print(seg_model.cfg)
#     seg_model.run_experiment()
    

    


