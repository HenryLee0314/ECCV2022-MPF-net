import utils.frame_utils as frame_utils
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
from os.path import *
from cv2 import imread
from math import ceil
import cv2


class CylinderCity200R(data.Dataset):
    def __init__(self, root = ''):

        flow_root_0 = join(root, 'flow')
        image_root_0 = join(root, 'image')

        Image_Prefix = 'Frame_'
        Flow_Prefix = 'original_file_Optical_Flow_Frame_'

        self.flow_list = []
        self.image_list = []

        for index in range(217):

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

        # img1 = frame_utils.read_gen(self.image_list[index][0][0])
        # img2 = frame_utils.read_gen(self.image_list[index][0][1])
        flow = frame_utils.read_gen(self.flow_list[index][0])
        # print("=================flow.shape=====================\n\n")
        # print(flow.shape)
        _h = flow.shape[0]
        _w = flow.shape[1]
        flow = cv2.resize(flow, (int(_w), int(3/4*_w)), interpolation = cv2.INTER_AREA)
        # print("=================flow.shape=====================\n\n", _h, _w)


        im_all = [imread(img) for img in [self.image_list[index][0][0], self.image_list[index][0][1]]]
        # print("=================im_all.shape=====================\n\n")
        # print(im_all[0].shape)
        im_all[0] = cv2.resize(im_all[0], (int(_w), int(3/4*_w)), interpolation = cv2.INTER_AREA)
        im_all[1] = cv2.resize(im_all[1], (int(_w), int(3/4*_w)), interpolation = cv2.INTER_AREA)
        # cv2.imshow("aaaa", im_all[0])
        # cv2.waitKey(0)
        # print("=================im_all.shape=====================\n\n")

        im_all = [im[:, :, :3] for im in im_all]

         # rescale the image size to be multiples of 64
        divisor = 64.
        H = im_all[0].shape[0]
        W = im_all[0].shape[1]

        H_ = int(ceil(H/divisor) * divisor)
        W_ = int(ceil(W/divisor) * divisor)
        for i in range(len(im_all)):
	        im_all[i] = cv2.resize(im_all[i], (W_, H_))

        for _i, _inputs in enumerate(im_all):
	        im_all[_i] = im_all[_i][:, :, ::-1]
	        im_all[_i] = 1.0 * im_all[_i]/255.0
	
	        im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
	        im_all[_i] = torch.from_numpy(im_all[_i])
	        im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
	        im_all[_i] = im_all[_i].float()
    
        im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)




        # images = [img1, img2]
        # images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        # image_0 = torch.from_numpy(images.astype(np.float32))
        flow_0 = torch.from_numpy(flow.astype(np.float32)).cuda()

        flow_0[1,:,:] = flow_0[1,:,:] * (3/4*_w) / _h 
        # print(flow_0.shape)
        # print(_h)

        return im_all, flow_0

    def __len__(self):
        return self.size