import torch
import torch.nn.functional as F

# import segmentation_models_pytorch.segmentation_models_pytorch as smp
import segmentation_models_pytorch as smp

from tqdm import tqdm
import numpy as np
import pandas as pd
import os, glob
import tifffile as tiff
from pathlib import Path
from imageio import imread, imsave
from prettyprinter import pprint
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

from astropy.visualization import PercentileInterval
interval_95 = PercentileInterval(95.0)

# progression path
transfer_dataset = {
                    'CAL_Creek': ['CAL_Creek_ref_20200928T18_S2', '20200926T01_ASC64'],
                    'AugustComplex': ['AugustComplex_ref_20200929T19_S2', '20200930T02_ASC35'],
                    'elephant': ['elephant_20171003T19_S2_optRef', '20170930T01_ASC64'],
                    'Sydney': ['Sydney_20191231T00_S2_optRef', '20191227T08_ASC9'],
                    'Chuckegg': ['Chuckegg_20190812T18_L8_optRef', '20190812T01_ASC20'],

                    # ##========> Sweden <=========
                    # 'Karbole': ['Karbole_20180808T10_S2_optRef', '20180809T05_DSC168'],
                    # 'Fajelsjo': ['20181007T10_S2.dNBR1', '20180809T05_DSC168'],
                    # 'Trangslet': ['20180831T10_S2.dNBR1', '20180809T05_DSC168'],
                    # 'Lillhardal': ['20181005T10_S2.dNBR1', '20180809T05_DSC168'],
                    # 'Doctor_Creek': ['20201003T18_S2.dNBR1', '20201001T13_DSC144'],
                    # 'FraserIsland': ['xx', 'xx'],

                    ##========> BC Wildfire <=========
                    # 'G90363': ['xx', 'xx'],
                    # 'R91233': ['xx', 'xx'],
                    # 'ChristieMountain': ['xx', 'xx'],
                    # 'TalbottCreek': ['xx', 'xx'],
            }

# A3_Opt_Reference_Masks
testing_dataset = {
                    'Sydney': ['Sydney_20191231T00_S2_optRef', '20191227T08_ASC9'],
                    'elephant': ['elephant_20171003T19_S2_optRef', '20170930T01_ASC64'],
                    'Carr': ["Carr_20181030T19_S2_optRef", "Carr_20181109T14_DSC115"],
                    'Thomas': ['Thomas_20171228T18_S2_optRef', '20171228T01_ASC137'],

                    'Mendocino': ["Mendocino_20180831T19_S2_optRef", "Mendocino_20180910T14_DSC115"],
                    'Eyik': ["Eyik_20180912T03_S2_optRef", "Eyik_20180914T22_DSC91"],

                    'AZ_Bush': ['AZ_Bush_20200628T18_S2_optRef', '20200625T13_DSC27'],
                    'Amazon': ["Amazon_20191116T14_S2_optRef", "Amazon_20191117T09_DSC39"],
                    'Zhigansk': ["Zhigansk_20190827T03_S2_optRef", "Zhigansk_20190913T22_DSC149"],
                    'AU_Nowra': ["AU_Nowra_20191231T00_S2_optRef", "AU_Nowra_20200105T19_DSC147"],
                    
                    # 'KolymaRiver': ['KolymaRiver_20200726T01_S2_optRef', '20200721T20_DSC148'],
                    # 'Morkoka': ["Morkoka_20190913T04_S2_optRef", "Morkoka_20190907T22_DSC62"],
                    
                    # # BC Wildfires
                    # 'C10970': ['C10970_20170904T19_S2_optRef', 'C10970_20170907T14_DSC86'],
                    # 'C50744': ['C50744_20170928T19_L8_optRef', 'C50744_20170924T14_DSC159'],
                    # 'G41607': ['G41607_20180929T19_S2_optRef', 'G41607_20180919T14_DSC159'],
                    # 'G80340': ['G80340_20180726T19_S2_optRef', 'G80340_20180621T01_ASC64'],
                    # 'G82215': ['G82215_20180921T19_S2_optRef', 'G82215_20180828T14_DSC13'],
                    # 'K20637': ['K20637_20171003T19_S2_optRef', 'K20637_20170930T01_ASC64'],
                    # 'N21628': ['N21628_20170927T18_L8_optRef', 'N21628_20170930T13_DSC71'],
                    # 'R11498': ['R11498_20180929T19_S2_optRef', 'R11498_20180912T02_ASC137'],
                    # 'R12068': ['R12068_20181004T19_S2_optRef', 'R12068_20180831T14_DSC57'],
                    # 'R21721': ['R21721_20180922T19_L8_optRef', 'R21721_20181006T14_DSC57'],
                    # 'R91947': ['R91947_20180918T19_L8_optRef', 'R91947_20180922T02_ASC108'],
                    # 'R92033': ['R92033_20180915T19_S2_optRef', 'R92033_20180824T14_DSC130'],
                    # 'VA1787': ['VA1787_20180929T19_S2_optRef', 'VA1787_20180907T14_DSC159'],
                    # 'C50647': ['C50647_20171003T19_S2_optRef', 'C50647_20170918T01_ASC64']

                   
            }

def zero_padding(arr, patchsize):
    # print("zero_padding patchsize: {}".format(patchsize))
    (h, w, c) = arr.shape
    pad_h = (1 + np.floor(h/patchsize)) * patchsize - h
    pad_w = (1 + np.floor(w/patchsize)) * patchsize - w

    arr_pad = np.pad(arr, ((0, int(pad_h)), (0, int(pad_w)), (0, 0)), mode='symmetric')
    return arr_pad

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_folder = Path(cfg.data_folder)

        self.trainPath = self.data_folder / "Historical_Wildfire_Dataset" 
        self.transferPath = self.data_folder / "Temporal_Progressions_Data" 
        
        self.modelPath = Path(cfg.modelPath)
        self.expmPath = Path(str(os.path.split(self.modelPath)[0]))
        self.model = torch.load(str(self.modelPath / "best_model.pth"))

        # seg_model = SegModel(cfg)
        # self.model = seg_model.model.load_state_dict(torch.load(str(os.path.split(self.modelPath)[0])))
        
        self.input_folder =  cfg.input_folder
        self.transfer_folder = cfg.transfer_folder

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.cfg.ENCODER, self.cfg.ENCODER_WEIGHTS)
        
        
    def inference(self, url, savePath):
        # model.cpu()
        patchsize = 512

        if '.tif' == str(url)[-4:]: # tif
            img0 = tiff.imread(url)
            img = interval_95(np.nan_to_num(img0, 0)) * 255
        elif '.png' == str(url)[-4:]: # png
            img = imread(url)


        input_patchsize = 2 * patchsize
        padSize = int(patchsize/2)

        H, W, C = img.shape
        img_pad0 = zero_padding(img, patchsize) # pad img into a shape: (m*PATCHSIZE, n*PATCHSIZE)
        img_pad = np.pad(img_pad0, ((padSize, padSize), (padSize, padSize), (0, 0)), mode='symmetric')

        img_preprocessed = self.preprocessing_fn(img_pad)
        in_tensor = torch.from_numpy(img_preprocessed.transpose(2, 0, 1)).unsqueeze(0)

        (Height, Width, Channels) = img_pad.shape
        pred_mask_pad = np.zeros((Height, Width))
        for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
            for j in range(0, Width - input_patchsize + 1, patchsize):
                # print(i, i+input_patchsize, j, j+input_patchsize)

                inputPatch = in_tensor[..., i:i+input_patchsize, j:j+input_patchsize]

                if self.cfg.ARCH == 'FCN':
                    predPatch = self.model(inputPatch.type(torch.cuda.FloatTensor))['out']
                else:
                    predPatch = self.model(inputPatch.type(torch.cuda.FloatTensor))

                predPatch = predPatch.squeeze().cpu().detach().numpy()#.round()
                pred_mask_pad[i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predPatch[padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify

        pred_mask = pred_mask_pad[padSize:padSize+H, padSize:padSize+W] # clip back to original shape

        if True:
            dataName = os.path.split(url)[-1][:-4]
            # imsave(SAVEPATH / f'{dataName}_input.png', (img).astype(np.uint8))
            # imsave(savePath / f"conf_{dataName}.png", (pred_mask*255).astype(np.uint8))
            imsave(savePath / f"{dataName}.png", (pred_mask.round()*255).astype(np.uint8))

        if False:
            plt.figure(figsize=(10,10))
            plt.subplot(121)
            # plt.imshow(interval_95(img_preprocessed[:H, :W]), cmap='gray')
            plt.imshow(interval_95(img), cmap='gray')
            plt.subplot(122)
            plt.imshow(pred_mask, cmap='gray')
            
        return pred_mask

    def compute_test_accuarcy(self, pred, ref, pntAccFlag, fireName):
        pred_torch = torch.from_numpy(pred) 
        ref_torch = torch.from_numpy(ref)

        pred_vec = pred_torch.view(-1,1)
        ref_vec = ref_torch.view(-1,1)

        # pntAccFlag = True
        if pntAccFlag:
            pos_index =  np.where(ref_vec==1)[0]
            neg_index =  np.where(ref_vec==0)[0]
            num_sampled_pnts = int(1e4)
            pnts = (list(pos_index[np.random.randint(0, len(pos_index), num_sampled_pnts)]) + 
                list(neg_index[np.random.randint(0, len(neg_index), num_sampled_pnts)]))
            pred_vec = pred_vec[pnts, 0]
            ref_vec = ref_vec[pnts, 0]

        F1 = (smp.utils.metrics.Fscore()(pred_vec, ref_vec)).cpu().detach().numpy()
        P = (smp.utils.metrics.Precision()(pred_vec, ref_vec)).cpu().detach().numpy()
        R = (smp.utils.metrics.Recall()(pred_vec, ref_vec)).cpu().detach().numpy()
        IoU = (smp.utils.metrics.IoU()(pred_vec, ref_vec)).cpu().detach().numpy()
        kappa = cohen_kappa_score(ref_vec, pred_vec, labels=np.unique(ref))
        OA = accuracy_score(ref_vec, pred_vec)

        print("\n---------------------------------------")
        print(f"Precision: {P}")
        print(f"Recall: {R}")
        print(f"OA: {OA}")
        print(f"Kappa: {kappa}")
        print(f"F1 score: {F1}")
        print(f"IoU: {IoU}")
        print("-----------------------------------------")

        Mat = np.array([self.cfg.ARCH, self.cfg.ENCODER, fireName, pntAccFlag, self.cfg.alpha, self.cfg.beta, P, R, OA, kappa, F1, IoU]).reshape(1,-1)
        df = pd.DataFrame(Mat, 
                        columns=["ARCH", "ENCODER", "fireName", 'pntAccFlag', 'alpha', 'beta', 'P', 'R', 'OA', 'Kappa', 'F1', 'IoU'], 
                        index=[f"{os.path.split(self.modelPath)[-1]}"])

        if pntAccFlag: csvName = f"transfer_accuracy_pntAcc.csv"
        else: csvName = f"transfer_accuracy.csv"

        if os.path.isfile(self.expmPath / csvName):
            df.to_csv(self.expmPath / csvName, mode='a', header=False) 
        else:
            df.to_csv(self.expmPath / csvName, mode='a')
        
        
    def model_testing(self):
        testSavePath = self.modelPath / f"testing_results"
        if not os.path.exists(testSavePath): os.mkdir(testSavePath)

        for fireName in testing_dataset.keys():
            print(f"fireName: {fireName}")
            refName = testing_dataset[fireName][0]
            ref = imread(self.trainPath / f"A3_Opt_Reference_Masks" / f"{refName}.png") / 255

            url = glob.glob(str(self.trainPath / self.input_folder / f"{fireName}*"))[0]
            dataName = os.path.split(url)[-1][:-4]
            # print(f"{file}")
            # -------------> model inference <--------------------- 
            pred_mask = self.inference(url, testSavePath)
            pred_mask_bin = pred_mask.round()

            print(f"{testing_dataset[fireName][1]} <--> {dataName}")
            if testing_dataset[fireName][1] in dataName:      
                print(f"{testing_dataset[fireName][1]} <--> {dataName}")  
                print("testing accuracy")
                self.compute_test_accuarcy(pred_mask_bin, ref, False, fireName)
                self.compute_test_accuarcy(pred_mask_bin, ref, True, fireName)


    def model_transfer(self):   
        for fireName in transfer_dataset.keys():
            print(f"transfer to {fireName}")             
            firePath = self.transferPath / f"{fireName}_Progression_Data_20m"
            
            refName = transfer_dataset[fireName][0]
            ref_url = firePath / "A0_Opt_Ref_Mask" / f"{refName}.png"
            if os.path.isfile(ref_url):
                ref = imread(ref_url) / 255.0
                print(ref.shape)
                if len(ref.shape) > 2: ref = ref[:,:,0]

            transferSavePath = self.modelPath / f"transfer_to_{fireName}"
            if not os.path.exists(transferSavePath): os.mkdir(transferSavePath)
            transferDataPath =  firePath / self.transfer_folder
            pprint(os.listdir(transferDataPath))
            for dataName in sorted(os.listdir(transferDataPath)):
                url = transferDataPath / dataName
                # print(f"{file}")
                # -------------> model inference <--------------------- 
                pred_mask = self.inference(url, transferSavePath)
                pred_mask_bin = pred_mask.round()

                if transfer_dataset[fireName][1] in dataName:        
                    # print("test accuracy")
                    self.compute_test_accuarcy(pred_mask_bin, ref, False, fireName)
                    self.compute_test_accuarcy(pred_mask_bin, ref, True, fireName)


if __name__ == "__main__":
    
    evalator = Evaluator(cfg)
    evalator.model_testing()
    evalator.model_transfer()