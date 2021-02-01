import torch
from easydict import EasyDict as edict
from datasets.data_samping import DataSampler
from pathlib import Path
from prettyprinter import pprint
from imageio import imread, imsave
import os, json


# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')

# if __name__ == '__main__':
#     run()

from config.configuration import cfg
cfg0 = cfg

project_dir = Path("/content/drive/My Drive/A_Wildfire_Projects/EO4Wildfire_Project")
Experiments_dir = project_dir / f"BC_Wildfire_Data" / "Experiments"
# Experiments_dir = project_dir / f"Data" / "Experiments"

# project_dir / f"BC_Wildfire_Data" / "Experiments", 
for Experiments_dir in [project_dir / f"Experiments_TV"]:
    for folder in sorted(list(os.listdir(Experiments_dir))):
        print(folder)
    # for folder in [
    #         # "20210110T180724_SAR_UNet_resnet50_a_0.5_b_1e-05_trainSize_3072",
    #         # "20210110T190302_OptSAR_UNet_resnet50_a_0.5_b_1e-05_trainSize_3072",
    #         # "20210110T183521_Opt_UNet_resnet50_a_0.5_b_1e-05_trainSize_3072",

    #         # "20210111T092915_SAR_UNet_resnet50_a_0.5_b_2e-05_trainSize_3072",
    #         # "20210111T104019_OptSAR_UNet_resnet50_a_0.5_b_2e-05_trainSize_3072",
    #         # "20210111T114659_Opt_UNet_resnet50_a_0.5_b_2e-05_trainSize_3072",

    #         # "20210112T152131_Opt_UNet_resnet50_a_0.5_b_2e-05_trainSize_3072"
    #         # "20210113T092526_SAR_UNet_resnet50_a_0.5_b_0_trainSize_3072",
    #         # "20210113T105105_OptSAR_UNet_resnet50_a_0.5_b_0_trainSize_3072",
    #         # "20210113T112231_Opt_UNet_resnet50_a_0.5_b_0_trainSize_3072",

    #         "20210118T083548_SARREF_UNet_resnet50_a_0.5_b_2e-05_trainSize_3072",
    #         "20210118T095934_OptSAR_UNet_resnet50_a_0.5_b_2e-05_trainSize_3072",
    #         "20210118T102525_OptREF_UNet_resnet50_a_0.5_b_2e-05_trainSize_3072"
    #     ]:

        model_url = str(Experiments_dir / folder)
        # model_url = str(project_dir / "Experiments/20201215T224831_SAR_UNet_resnet50_a_0.5_b_1e-05_trainSize_2048")

        """ load configuration """
        cfg_url = os.path.join(model_url, 'config.json')
        fp = open (cfg_url, "r")
        cfg = edict(json.load(fp))

        cfg.input_folder = "A0_SAR_Input_PNG"
        cfg.transfer_folder = "A0_Progression_PNG"
        cfg.data_folder = cfg0.data_folder

        cfg.modelPath = model_url

        pprint(cfg)
        # pprint(cfg.rootPath)
        
        """ Evaluation """
        # # specify the data_folder for model testing
        from evaluation.BC_evaluator import Evaluator
        cfg.data_folder = str(Path(os.getcwd()) / "BC_Wildfire_Data")
        evaluator = Evaluator(cfg)
        # evaluator.model_testing()

        # specify the data_folder for model transfering
        from evaluation.BC_evaluator import Evaluator
        cfg.data_folder = str(Path(os.getcwd()) / "Data")
        evaluator = Evaluator(cfg)
        evaluator.model_transfer()

