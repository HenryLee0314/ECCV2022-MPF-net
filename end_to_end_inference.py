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

import sys
sys.path.append('.')

import libexample


from cv2 import imread
from math import ceil
import subprocess, shutil

from datasets.fusion_city100_r import FusionCity100R
from datasets.end_to_end_dataset import E2E_City100R


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

def flow_cubepadding_to_equirect(flow):
    _SIZE_ = 256
    front = flow[2*_SIZE_:3*_SIZE_, 2*_SIZE_:3*_SIZE_, :]
    back = flow[2*_SIZE_:3*_SIZE_, 0:_SIZE_, :]
    left = flow[2*_SIZE_:3*_SIZE_, 1*_SIZE_:2*_SIZE_, :]
    right = flow[2*_SIZE_:3*_SIZE_, 3*_SIZE_:4*_SIZE_, :]
    top = flow[1*_SIZE_:2*_SIZE_, 2*_SIZE_:3*_SIZE_, :]
    bottom = flow[3*_SIZE_:4*_SIZE_, 2*_SIZE_:3*_SIZE_, :]
    result = libexample.flow_cubemap_to_equirect(front.astype(np.float32), back.astype(np.float32), left.astype(np.float32), right.astype(np.float32), top.astype(np.float32), bottom.astype(np.float32), flow.shape[0], flow.shape[1], flow.shape[2])
    return result.astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--testing_dataset', type=str, default="/media/yiheng/T2/__DATASET__/City_100_r", help="training dataset path")
    parser.add_argument('--E_C_fusion_model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/CIty_E_C_from_Mar_6/PWC_model_best.pth.tar', help='model path')
    parser.add_argument('--E_P_fusion_model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/CIty_E_P_from_Feb_26/PWC_model_best.pth.tar', help='model path')
    parser.add_argument('--C_P_fusion_model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/City_C_P_from_Feb_28/PWC_model_best.pth.tar', help='model path')
    parser.add_argument('--equirect_model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/City_E_from_Nov_17/PWC_model_best.pth.tar', help='model path')
    parser.add_argument('--cylinder_model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/City_C_from_Nov_28/PWC_model_best.pth.tar', help='model path')
    parser.add_argument('--cubepadding_model_path', type=str, default='/home/yiheng/project/ECCV2022-multi-proj-optical-flow/pretrain_models/City_P_from_Feb_26/PWC_model_best.pth.tar', help='model path')
    parser.add_argument('--enable_equirect', type=bool, default=True, help="enable equirect")
    parser.add_argument('--enable_cylinder', type=bool, default=True, help="enable cylinder")
    parser.add_argument('--enable_cubepadding', type=bool, default=False, help="enable cubepadding")
    parser.add_argument('--enable_fusion_E_C', type=bool, default=True, help="enable fusion_E_C")
    parser.add_argument('--enable_fusion_E_P', type=bool, default=False, help="enable fusion_E_P")
    parser.add_argument('--enable_fusion_C_P', type=bool, default=False, help="enable fusion_C_P")


    args = parser.parse_args()

    test_dataset = E2E_City100R(args.testing_dataset)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    progress = tqdm(test_loader, ncols=100, total=len(test_loader))

    equirect_model = pwc_dc_net(args.equirect_model_path)
    equirect_model.cuda()
    equirect_model.eval()

    cylinder_model = pwc_dc_net(args.cylinder_model_path)
    cylinder_model.cuda()
    cylinder_model.eval()

    cubepadding_model = pwc_dc_net(args.cubepadding_model_path)
    cubepadding_model.cuda()
    cubepadding_model.eval()

    E_C_fusion_model = get_fusion_net(args.E_C_fusion_model_path)
    E_C_fusion_model.cuda()
    E_C_fusion_model.eval()

    E_P_fusion_model = get_fusion_net(args.E_P_fusion_model_path)
    E_P_fusion_model.cuda()
    E_P_fusion_model.eval()

    C_P_fusion_model = get_fusion_net(args.C_P_fusion_model_path)
    C_P_fusion_model.cuda()
    C_P_fusion_model.eval()


    for batch_idx, (equirect_RGB, equirect_flow, cylinder_RGB, cylinder_flow, cubepadding_RGB, cubepadding_flow) in enumerate(progress):
        if (args.enable_equirect):
            equirect_output = equirect_model(equirect_RGB[:,0,:,:,:].cuda())
            equirect_output = equirect_output[0][0].cpu().data.numpy() # batch size must be 1
            equirect_output = equirect_output * 20.0

            equirect_output = equirect_output
            equirect_output = np.swapaxes(np.swapaxes(equirect_output, 0, 1), 1, 2) # 
            W = 1024
            H = 512
            u_ = cv2.resize(equirect_output[:,:,0],(W,H))
            v_ = cv2.resize(equirect_output[:,:,1],(W,H))
            equirect_output = np.dstack((u_,v_))
            # writeFlowFile("tmp/%06d"%batch_idx + '.flo', equirect_output)


        if (args.enable_cylinder):
            cylinder_output = cylinder_model(cylinder_RGB[:,0,:,:,:].cuda())
            cylinder_output = cylinder_output[0][0].cpu().data.numpy() # batch size must be 1
            cylinder_output = cylinder_output * 20.0

            cylinder_output = np.swapaxes(np.swapaxes(cylinder_output, 0, 1), 1, 2)
            W = 1024
            H = 2706
            u_ = cv2.resize(cylinder_output[:,:,0],(W,H))
            v_ = cv2.resize(cylinder_output[:,:,1],(W,H))
            v_ = v_ * H / (3/4*W)
            cylinder_output = np.dstack((u_,v_))

            cylinder_output = libexample.flow_cylinder_to_equirect(cylinder_output.astype(np.float32), cylinder_output.shape[0], cylinder_output.shape[1], cylinder_output.shape[2])
            # writeFlowFile("tmp/%06d"%batch_idx + '.flo', cylinder_output)

        if (args.enable_cubepadding):
            cubepadding_output = cubepadding_model(cubepadding_RGB[:,0,:,:,:].cuda())
            cubepadding_output = cubepadding_output[0][0].cpu().data.numpy() # batch size must be 1
            cubepadding_output = cubepadding_output * 20.0
            cubepadding_output = np.swapaxes(np.swapaxes(cubepadding_output, 0, 1), 1, 2) # 
            W = 1024
            H = 1024
            u_ = cv2.resize(cubepadding_output[:,:,0],(W,H))
            v_ = cv2.resize(cubepadding_output[:,:,1],(W,H))
            cubepadding_output = np.dstack((u_,v_))

            cubepadding_output = flow_cubepadding_to_equirect(cubepadding_output)
            # writeFlowFile("tmp/%06d"%batch_idx + '.flo', cubepadding_output)


        if (args.enable_fusion_E_C):
            equirect_output = equirect_output.transpose(2,0,1)
            cylinder_output = cylinder_output.transpose(2,0,1)

            equirect_output = torch.from_numpy(equirect_output.astype(np.float32))
            cylinder_output = torch.from_numpy(cylinder_output.astype(np.float32))

            equirect_output = equirect_output.expand(1, equirect_output.size()[0], equirect_output.size()[1], equirect_output.size()[2])
            cylinder_output = equirect_output.expand(1, cylinder_output.size()[0], cylinder_output.size()[1], cylinder_output.size()[2])

            equirect_output = torch.autograd.Variable(equirect_output.cuda(), volatile=True)
            cylinder_output = torch.autograd.Variable(cylinder_output.cuda(), volatile=True)

            fusion_RGB = equirect_RGB[0,:,0:3, :, :]
            E_C_fusion_output = E_C_fusion_model(equirect_output, cylinder_output, fusion_RGB * 255)

            E_C_fusion_output = E_C_fusion_output[0].cpu().data.numpy()
            E_C_fusion_output = np.swapaxes(np.swapaxes(E_C_fusion_output, 0, 1), 1, 2)
            W = 1024
            H = 512
            u_ = cv2.resize(E_C_fusion_output[:,:,0],(W,H))
            v_ = cv2.resize(E_C_fusion_output[:,:,1],(W,H))
            E_C_fusion_output = np.dstack((u_,v_))
            writeFlowFile("tmp/%06d"%batch_idx + '.flo', E_C_fusion_output)

        if (args.enable_fusion_E_P):
            equirect_output = equirect_output.transpose(2,0,1)
            cubepadding_output = cubepadding_output.transpose(2,0,1)

            equirect_output = torch.from_numpy(equirect_output.astype(np.float32))
            cubepadding_output = torch.from_numpy(cubepadding_output.astype(np.float32))

            equirect_output = equirect_output.expand(1, equirect_output.size()[0], equirect_output.size()[1], equirect_output.size()[2])
            cubepadding_output = equirect_output.expand(1, cubepadding_output.size()[0], cubepadding_output.size()[1], cubepadding_output.size()[2])

            equirect_output = torch.autograd.Variable(equirect_output.cuda(), volatile=True)
            cubepadding_output = torch.autograd.Variable(cubepadding_output.cuda(), volatile=True)

            fusion_RGB = equirect_RGB[0,:,0:3, :, :]
            E_P_fusion_output = E_P_fusion_model(equirect_output, cubepadding_output, fusion_RGB * 255)

            E_P_fusion_output = E_P_fusion_output[0].cpu().data.numpy()
            E_P_fusion_output = np.swapaxes(np.swapaxes(E_P_fusion_output, 0, 1), 1, 2)
            W = 1024
            H = 512
            u_ = cv2.resize(E_P_fusion_output[:,:,0],(W,H))
            v_ = cv2.resize(E_P_fusion_output[:,:,1],(W,H))
            E_P_fusion_output = np.dstack((u_,v_))
            # writeFlowFile("tmp/%06d"%batch_idx + '.flo', E_P_fusion_output)

        if (args.enable_fusion_C_P):
            cylinder_output = cylinder_output.transpose(2,0,1)
            cubepadding_output = cubepadding_output.transpose(2,0,1)

            cylinder_output = torch.from_numpy(cylinder_output.astype(np.float32))
            cubepadding_output = torch.from_numpy(cubepadding_output.astype(np.float32))

            cylinder_output = cylinder_output.expand(1, cylinder_output.size()[0], cylinder_output.size()[1], cylinder_output.size()[2])
            cubepadding_output = cylinder_output.expand(1, cubepadding_output.size()[0], cubepadding_output.size()[1], cubepadding_output.size()[2])

            cylinder_output = torch.autograd.Variable(cylinder_output.cuda(), volatile=True)
            cubepadding_output = torch.autograd.Variable(cubepadding_output.cuda(), volatile=True)

            fusion_RGB = equirect_RGB[0,:,0:3, :, :]
            C_P_fusion_output = C_P_fusion_model(cylinder_output, cubepadding_output, fusion_RGB * 255)

            C_P_fusion_output = C_P_fusion_output[0].cpu().data.numpy()
            C_P_fusion_output = np.swapaxes(np.swapaxes(C_P_fusion_output, 0, 1), 1, 2)
            W = 1024
            H = 512
            u_ = cv2.resize(C_P_fusion_output[:,:,0],(W,H))
            v_ = cv2.resize(C_P_fusion_output[:,:,1],(W,H))
            C_P_fusion_output = np.dstack((u_,v_))
            # writeFlowFile("tmp/%06d"%batch_idx + '.flo', C_P_fusion_output)
    

        


