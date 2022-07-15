import argparse
import torch
# import torch.nn as nn
import torch.optim as optim

from numpy import sin, cos, tan, pi, arcsin, arctan
from torch import nn
from torchsummary import summary
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import os
from os.path import *

from glob import glob

import numpy as np

from tqdm import tqdm

import cv2

import torch.nn.functional as F

from models.PWCNet import pwc_dc_net
from models.fusion import get_fusion_net
import sys
import math
from torch.autograd import Variable
from cv2 import imread
import models
from models import *


from cv2 import imread
from math import ceil
import subprocess, shutil

from datasets.fusion_city100_r import FusionCity100R
# from datasets.fusion_EFT_car100 import FusionEFT100
# from datasets.fusion_city100 import FusionCity100

from datasets.fusion_city100_r_C_P import FusionCity100RCP
from datasets.fusion_EFT_car100_C_P import FusionEFT100CP

from tensorboardX import SummaryWriter




def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--testing_dataset', type=str, default="/media/yiheng/T2/__DATASET__/City_100_r", help="training dataset path")
    parser.add_argument('--model_path', type=str, default='/media/yiheng/T2/2022_Feb_28_C_P_models/work_fusion_1_City/PWC_model_best.pth.tar', help='model path')

    parser.add_argument('--model', type=str, default='PWC', help='model')



    args = parser.parse_args()


    test_dataset = FusionCity100RCP(args.testing_dataset)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    progress = tqdm(test_loader, ncols=100, total=len(test_loader))




    fusion_model = get_fusion_net(args.model_path)
    fusion_model.cuda()
    fusion_model.eval()

    

    for batch_idx, (data_i, target, data_e, data_c) in enumerate(progress):
        output = fusion_model(data_e, data_c, data_i)

        output = output[0].cpu().data.numpy() # batch size must be 1
        print(output.shape)

        # scale the flow back to the input size 
        output = np.swapaxes(np.swapaxes(output, 0, 1), 1, 2) # 
        W = 1024
        H = 512
        u_ = cv2.resize(output[:,:,0],(W,H))
        v_ = cv2.resize(output[:,:,1],(W,H))
        output = np.dstack((u_,v_))
        # print(output.shape)
        writeFlowFile("tmp/%06d"%batch_idx + '.flo', output)
    

        


