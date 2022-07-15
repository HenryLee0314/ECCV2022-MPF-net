import argparse
import torch

from torch.utils.data import DataLoader
import os
from os.path import *

import numpy as np

from tqdm import tqdm

import cv2

from models.PWCNet import pwc_dc_net
import sys
import math
import models
from models import *

from datasets.cylinder_city100_r import CylinderCity100R
from datasets.cylinder_city100 import CylinderCity100
from datasets.cylinder_city2000_r import CylinderCity2000R
from datasets.cylinder_city200_r import CylinderCity200R
from datasets.cylinder_EFT_car2000 import Cylinder_EFT_CAR2000
from datasets.cylinder_EFT_car200 import Cylinder_EFT_CAR200
from datasets.cylinder_EFT_car100 import Cylinder_EFT_CAR100
from math import ceil
import subprocess, shutil


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
    parser.add_argument('--effective_batch_size', '-e', type=int, default=1, help="effective_batch_size")

    # parser.add_argument('--testing_dataset', type=str, default="/media/yiheng/T2/__DATASET__/City_Cylinder_100_R", help="teating dataset path")
    parser.add_argument('--testing_dataset', type=str, default="/media/yiheng/T2/__DATASET__/EFTs_Cylinder_Car100", help="teating dataset path")

    # parser.add_argument('--model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/City_C_from_Nov_28/PWC_model_best.pth.tar', help='model path')
    parser.add_argument('--model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/EFT_C_from_Nov_26/PWC_model_best.pth.tar', help='model path')


    parser.add_argument('--model', type=str, default='PWC', help='model')
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')

    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()


 
    test_dataset = Cylinder_EFT_CAR100(args.testing_dataset)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    progress = tqdm(test_loader, ncols=100, total=len(test_loader))




    model = pwc_dc_net(args.model_path)
    model.cuda()
    model.eval()

    for batch_idx, (input, gt_flow) in enumerate(progress):
        output = model(input[:,0,:,:,:].cuda())

        output = output[0][0].cpu().data.numpy() # batch size must be 1
        output = output * 20.0

        # scale the flow back to the input size 
        output = np.swapaxes(np.swapaxes(output, 0, 1), 1, 2) # 
        W = 1024
        H = 2706
        u_ = cv2.resize(output[:,:,0],(W,H))
        v_ = cv2.resize(output[:,:,1],(W,H))
        v_ = v_ * H / (3/4*W)

        output = np.dstack((u_,v_))
        # print(output.shape)
        writeFlowFile("tmp/%06d"%batch_idx + '.flo', output)

        


