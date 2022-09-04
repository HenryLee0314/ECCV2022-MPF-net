import torch
# import torch.nn as nn
import torch.optim as optim

from numpy import sin, cos, tan, pi, arcsin, arctan
from torch import nn
import os
import numpy as np
import cv2

TAG_FLOAT = 202021.25

def read_flow(file):

	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	# print("w: ", w)
	h = np.fromfile(f, np.int32, count=1)
	# print("h: ", h)
	#if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	# data = np.fromfile(f, np.float32, count=2*w*h)
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h), int(w), 2))	
	f.close()

	return flow

from functools import lru_cache
import math
@lru_cache(None)
def generate_weight(H, W):
    weight = np.zeros([1, H, W])

    test = 0
    _max = 0
    for h in range(H):
        for w in range (W):
            theta = ((w+0.5) / W * 2 - 1) * math.pi
            phi = ((h+0.5) / H * 2 - 1) * math.pi / 2

            right_theta = ((w+0.5 + 1) / W * 2 - 1) * math.pi
            right_phi = phi

            down_theta = theta
            down_phi = ((h+0.5+1) / H * 2 - 1) * math.pi / 2

            _x = math.cos(phi) * math.cos(theta)
            _y = math.sin(phi)
            _z = math.cos(phi) * math.sin(theta)

            right_x = math.cos(right_phi) * math.cos(right_theta)
            right_y = math.sin(right_phi)
            right_z = math.cos(right_phi) * math.sin(right_theta)

            down_x = math.cos(down_phi) * math.cos(down_theta)
            down_y = math.sin(down_phi)
            down_z = math.cos(down_phi) * math.sin(down_theta)

            # print(math.sqrt(_x*_x+_y*_y+_z*_z))

            to_right_x = right_x - _x
            to_right_y = right_y - _y
            to_right_z = right_z - _z

            to_down_x = down_x - _x
            to_down_y = down_y - _y
            to_down_z = down_z - _z

            cross_x = to_right_y*to_down_z-to_right_z*to_down_y
            cross_y = to_right_z*to_down_x-to_right_x*to_down_z
            cross_z = to_right_x*to_down_y-to_right_y*to_down_x

            ret = math.sqrt(cross_x*cross_x+cross_y*cross_y+cross_z*cross_z)
            # print(ret)
            test += ret
            if (ret > _max):
                _max = ret

            weight[0,h,w] = ret
    # print(test, 4*math.pi)
    return weight, _max

equirect_weight_map, weight_map_max = generate_weight(512, 1024)
equirect_weight_map =  equirect_weight_map / weight_map_max


equirect_weight_map = torch.FloatTensor(equirect_weight_map).cuda()


def EPE(input_flow, target_flow):
    ret = torch.norm(target_flow-input_flow, p=2, dim=0)
    # ret = ret * equirect_weight_map
    ret = ret.mean()
    return ret

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output-target, p=2, dim=1).mean()
        return lossvalue

def calculate_EPE(flow_gt, flow_target):
	x = flow_target[:,:,0] - flow_gt[:,:,0]
	y = flow_target[:,:,1] - flow_gt[:,:,1]
	d = np.sqrt(x ** 2 + y ** 2)
	return d, (np.sum(d) / (512*1024)), np.max(d)


class DataGroup():
    def __init__(self, name = '', root = ''):
        self.name = name
        self.root = root
        self.AEPE = 0
        self.ARAD = 0
        self.counter = 0
        self.grid = torch.FloatTensor(generate_grid(512, 1024)).cuda()
    
    def calculate(self, index, flow_data_gt):
        flow_file = self.root + "%06d"%(index) +'.flo'
        flow_data = read_flow(flow_file)
        flow_data = np.swapaxes(np.swapaxes(flow_data, 1, 2), 0, 1)
        flow_data = torch.FloatTensor(flow_data).cuda()
        rad = calculate_great_circle_distance(flow_data, flow_data_gt, self.grid).cpu().data.numpy()
        self.ARAD += rad
        epe = EPE(flow_data, flow_data_gt).cpu().data.numpy()
        self.AEPE += epe
        self.counter += 1
        print(self.name + ": index[" + str(index) + "]: ", epe , "[RAD]:", rad)

    def return_result(self):
        print(self.name + " [AEPE]: ", self.AEPE / self.counter, " [RAD]: ", self.ARAD / self.counter)


@lru_cache(None)
def generate_grid(H, W):
    weight = np.zeros([2, H, W])

    for h in range(H):
        for w in range (W):
            theta = ((w+0.5) / W * 2 - 1) * math.pi
            phi = ((h+0.5) / H * 2 - 1) * math.pi / 2
            weight[0,h,w] = theta
            weight[1,h,w] = phi

    return weight


def calculate_great_circle_distance(input_flow, target_flow, grid):
    input = torch.clone(input_flow)
    gt = torch.clone(target_flow)
    input[0, :, :] = input[0, :, :] / (1024/2) * 3.1415926
    input[1, :, :] = input[1, :, :] / (512/2) * 3.1415926
    gt[0, :, :] = gt[0, :, :] / (1024/2) * 3.1415926
    gt[1, :, :] = gt[1, :, :] / (512/2) * 3.1415926
    input = grid + input
    gt = grid + gt
    theta = input[0, :, :]
    phi = input[1, :, :]

    _x = torch.cos(phi) * torch.cos(theta)
    _y = torch.sin(phi)
    _z = torch.cos(phi) * torch.sin(theta)
    gt_theta = gt[0, :, :]
    gt_phi = gt[1, :, :] 

    gt_x = torch.cos(gt_phi) * torch.cos(gt_theta)
    gt_y = torch.sin(gt_phi)
    gt_z = torch.cos(gt_phi) * torch.sin(gt_theta)
    dot_result = _x * gt_x + _y * gt_y + _z * gt_z
    dot_result[dot_result<-1] = -1
    dot_result[dot_result>1] = 1
    angle = torch.arccos(dot_result)

    arc = angle * 1024.0 / (2 * 3.1415926)

    ret = arc.mean()
    return ret

tmp = DataGroup("tmp", "./tmp/")
# tmp = DataGroup("tmp", "/home/yiheng/project/EquirectProject/build/tmp/")




for index in range(100):
    if index == 0:
        continue

    flow_file_gt = "/media/yiheng/T2/__DATASET__/EFTs_Car100/flow/original_file_Optical_Flow_Frame_" + "%05d"%(index) +'.flo'
    flow_data_gt = read_flow(flow_file_gt)
    flow_data_gt = np.swapaxes(np.swapaxes(flow_data_gt, 1, 2), 0, 1)

    l2 = L2().cuda()
    l1 = L1().cuda()

    flow_data_gt = torch.FloatTensor(flow_data_gt).cuda()

    tmp.calculate(index-1, flow_data_gt)

tmp.return_result()
