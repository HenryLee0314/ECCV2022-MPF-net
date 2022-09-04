import utils.frame_utils as frame_utils
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
from os.path import *
from cv2 import imread
from math import ceil
import cv2

import sys
sys.path.append('.')
sys.path.append('..')

import libexample

class MPFDataset(data.Dataset):
    def __init__(self, root, size):

        flow_root_0 = join(root, 'flow')
        image_root_0 = join(root, 'image')

        Image_Prefix = 'Frame_'
        Flow_Prefix = 'original_file_Optical_Flow_Frame_'

        self.flow_list = []
        self.image_list = []

        for index in range(size):

            if index == 0:
                continue

            image_0_source = join(image_root_0, Image_Prefix + "%05d"%(index+0) + '.png')
            image_0_target = join(image_root_0, Image_Prefix + "%05d"%(index-1) + '.png')

            flow_0 = join(flow_root_0, Flow_Prefix + "%05d"%(index+0) + '.flo')

            if not isfile(image_0_source) or not isfile(image_0_target) or not isfile(flow_0):
                print(isfile(image_0_source), isfile(image_0_target), isfile(flow_0))
                continue

            image = [[image_0_source, image_0_target]]
            flow = [flow_0]

            self.image_list += [image]
            self.flow_list += [flow]

        self.size = len(self.image_list)

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):
        flow = frame_utils.read_gen(self.flow_list[index][0])


        equirect_RGB = [imread(img) for img in [self.image_list[index][0][0], self.image_list[index][0][1]]]
        equirect_RGB = [im[:, :, :3] for im in equirect_RGB]

         # rescale the image size to be multiples of 64
        divisor = 64.
        H = equirect_RGB[0].shape[0]
        W = equirect_RGB[0].shape[1]

        H_ = int(ceil(H/divisor) * divisor)
        W_ = int(ceil(W/divisor) * divisor)

        #convert E to C & P
        cylinder_RGB_0 = libexample.RGB_equirect_to_cylinder(equirect_RGB[0], equirect_RGB[0].shape[0], equirect_RGB[0].shape[1], equirect_RGB[0].shape[2])
        cylinder_RGB_1 = libexample.RGB_equirect_to_cylinder(equirect_RGB[1], equirect_RGB[1].shape[0], equirect_RGB[1].shape[1], equirect_RGB[1].shape[2])

        cylinder_flow = libexample.flow_equirect_to_cylinder(flow.astype(np.float32), flow.shape[0], flow.shape[1], flow.shape[2])
        cylinder_flow_h = cylinder_flow.shape[0]
        cylinder_flow_w = cylinder_flow.shape[1]

        cylinder_RGB_0 = cv2.resize(cylinder_RGB_0, (int(cylinder_flow_w), int(3/4*cylinder_flow_w)), interpolation = cv2.INTER_AREA)
        cylinder_RGB_1 = cv2.resize(cylinder_RGB_1, (int(cylinder_flow_w), int(3/4*cylinder_flow_w)), interpolation = cv2.INTER_AREA)
        
        cylinder_flow = cv2.resize(cylinder_flow, (int(cylinder_flow_w), int(3/4*cylinder_flow_w)), interpolation = cv2.INTER_AREA)

        cubepadding_RGB_0 = libexample.RGB_equirect_to_cubepadding(equirect_RGB[0], equirect_RGB[0].shape[0], equirect_RGB[0].shape[1], equirect_RGB[0].shape[2])
        cubepadding_RGB_1 = libexample.RGB_equirect_to_cubepadding(equirect_RGB[1], equirect_RGB[1].shape[0], equirect_RGB[1].shape[1], equirect_RGB[1].shape[2])

        cubepadding_flow = libexample.flow_equirect_to_cubepadding(flow.astype(np.float32), flow.shape[0], flow.shape[1], flow.shape[2])

        cylinder_RGB = [cylinder_RGB_0, cylinder_RGB_1]
        cubepadding_RGB = [cubepadding_RGB_0, cubepadding_RGB_1]

        for i in range(len(equirect_RGB)):
            equirect_RGB[i] = cv2.resize(equirect_RGB[i], (W_, H_))

        for _i, _ in enumerate(equirect_RGB):
            equirect_RGB[_i] = equirect_RGB[_i][:, :, ::-1]
            equirect_RGB[_i] = 1.0 * equirect_RGB[_i]/255.0
    
            equirect_RGB[_i] = np.transpose(equirect_RGB[_i], (2, 0, 1))
            equirect_RGB[_i] = torch.from_numpy(equirect_RGB[_i])
            equirect_RGB[_i] = equirect_RGB[_i].expand(1, equirect_RGB[_i].size()[0], equirect_RGB[_i].size()[1], equirect_RGB[_i].size()[2])    
            equirect_RGB[_i] = equirect_RGB[_i].float()

            # cylinder
            cylinder_RGB[_i] = cylinder_RGB[_i][:, :, ::-1]
            cylinder_RGB[_i] = 1.0 * cylinder_RGB[_i]/255.0
    
            cylinder_RGB[_i] = np.transpose(cylinder_RGB[_i], (2, 0, 1))
            cylinder_RGB[_i] = torch.from_numpy(cylinder_RGB[_i])
            cylinder_RGB[_i] = cylinder_RGB[_i].expand(1, cylinder_RGB[_i].size()[0], cylinder_RGB[_i].size()[1], cylinder_RGB[_i].size()[2])    
            cylinder_RGB[_i] = cylinder_RGB[_i].float()

            # cubepadding
            cubepadding_RGB[_i] = cubepadding_RGB[_i][:, :, ::-1]
            cubepadding_RGB[_i] = 1.0 * cubepadding_RGB[_i]/255.0
    
            cubepadding_RGB[_i] = np.transpose(cubepadding_RGB[_i], (2, 0, 1))
            cubepadding_RGB[_i] = torch.from_numpy(cubepadding_RGB[_i])
            cubepadding_RGB[_i] = cubepadding_RGB[_i].expand(1, cubepadding_RGB[_i].size()[0], cubepadding_RGB[_i].size()[1], cubepadding_RGB[_i].size()[2])    
            cubepadding_RGB[_i] = cubepadding_RGB[_i].float()

    
        with torch.no_grad():  
            equirect_RGB = torch.autograd.Variable(torch.cat(equirect_RGB,1).cuda())
            cylinder_RGB = torch.autograd.Variable(torch.cat(cylinder_RGB,1).cuda())
            cubepadding_RGB = torch.autograd.Variable(torch.cat(cubepadding_RGB,1).cuda())


        equirect_flow = flow.transpose(2,0,1)
        cylinder_flow = equirect_flow.transpose(2,0,1)
        cubepadding_flow = cubepadding_flow.transpose(2,0,1)

        equirect_flow = torch.from_numpy(equirect_flow.astype(np.float32))
        cylinder_flow = torch.from_numpy(cylinder_flow.astype(np.float32))
        cylinder_flow[1,:,:] = cylinder_flow[1,:,:] * (3/4*cylinder_flow_w) / cylinder_flow_h 
        cubepadding_flow = torch.from_numpy(cubepadding_flow.astype(np.float32))


        return equirect_RGB, equirect_flow, cylinder_RGB, cylinder_flow, cubepadding_RGB, cubepadding_flow

    def __len__(self):
        return self.size

