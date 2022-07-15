import utils.frame_utils as frame_utils
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
from os.path import *
from cv2 import imread
from math import ceil
import cv2

class FusionCity200RCP(data.Dataset):
    def __init__(self, root = ''):

        flow_root_0 = join(root, 'flow')
        image_root_0 = join(root, 'image')

        Image_Prefix = 'Frame_'
        Flow_Prefix = 'original_file_Optical_Flow_Frame_'

        self.flow_list = []
        self.image_list = []

        self.flow_e_list = []
        self.flow_c_list = []

        # flow_e_prefix = "/home/liyihe/Desktop/final_PWC/dataset_for_fusion/city/val/e/"
        # flow_c_prefix = "/home/liyihe/Desktop/final_PWC/dataset_for_fusion/city/val/c/"

        flow_e_prefix = "/home/liyihe/Desktop/___DATASET___/dataset_for_fusion/city/val/c/"
        flow_c_prefix = "/home/liyihe/Desktop/___DATASET___/__FUSION__/City200R_P/"

        for index in range(217):

            if index == 0:
                continue

            image_0_source = join(image_root_0, Image_Prefix + "%05d"%(index+0) + '.png')
            # image_0_target = join(image_root_0, Image_Prefix + "%05d"%(index-1) + '.png')

            flow_0 = join(flow_root_0, Flow_Prefix + "%05d"%(index+0) + '.flo')

            flow_e = flow_e_prefix + "%06d"%(index-1) + '.flo'
            flow_c = flow_c_prefix + "%06d"%(index-1) + '.flo'

            if not isfile(image_0_source) or not isfile(flow_0):
                print(isfile(image_0_source), isfile(flow_0))
                continue

            if not isfile(flow_e) or not isfile(flow_c):
                print(isfile(flow_e), isfile(flow_c))
                continue

            image = [image_0_source]
            flow = [flow_0]

            flow_e_temp = [flow_e]
            flow_c_temp = [flow_c]

            self.image_list += [image]
            self.flow_list += [flow]

            self.flow_e_list += [flow_e_temp]
            self.flow_c_list += [flow_c_temp]

        self.size = len(self.image_list)

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        flow = frame_utils.read_gen(self.flow_list[index][0])


        # im_all = [imread(img) for img in [self.image_list[index][0][0], self.image_list[index][0][1]]]
        image_src = imread(self.image_list[index][0])
        # print(image_src.shape)
        image_src = image_src[:, :, :3]
        image_src = np.transpose(image_src, (2, 0, 1))
        image_src = torch.from_numpy(image_src.astype(np.float32))
    
        image_src = torch.autograd.Variable(image_src.cuda(), volatile=True)


        flow = flow.transpose(2,0,1)
        flow_0 = torch.from_numpy(flow.astype(np.float32))


        flow_e = frame_utils.read_gen(self.flow_e_list[index][0])
        flow_e = flow_e.transpose(2,0,1)
        flow_e = torch.from_numpy(flow_e.astype(np.float32))
        flow_e = torch.autograd.Variable(flow_e.cuda(), volatile=True)

        flow_c = frame_utils.read_gen(self.flow_c_list[index][0])
        flow_c = flow_c.transpose(2,0,1)
        flow_c = torch.from_numpy(flow_c.astype(np.float32))
        flow_c = torch.autograd.Variable(flow_c.cuda(), volatile=True)


        return image_src, flow_0, flow_e, flow_c

    def __len__(self):
        return self.size