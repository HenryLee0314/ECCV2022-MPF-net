import sys
sys.path.append('.')

import os
import cv2
import numpy as np

import libexample

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

def calculate_EPE(flow_gt, flow_target):
    # assert flow_target.shape[0] is flow_gt.shape[0]
    x = flow_target[:,:,0] - flow_gt[:,:,0]
    y = flow_target[:,:,1] - flow_gt[:,:,1]
    d = np.sqrt(x ** 2 + y ** 2)
    # return np.sum(d) / (flow_target.shape[0] * flow_target.shape[1])
    return d.mean()

def calculate_MSE(RGB_gt, RGB_target):
    x = RGB_target[:,:,0] - RGB_gt[:,:,0]
    y = RGB_target[:,:,1] - RGB_gt[:,:,1]
    z = RGB_target[:,:,2] - RGB_gt[:,:,2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return d.mean()

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
    for index in range(2):
        if index == 0:
            continue

        image_path = '/home/yiheng/project/test_data/city100R_E/image/Frame_' + "%05d"%(index) +'.png'
        flow_file_gt = "/home/yiheng/project/test_data/city100R_E/flow/original_file_Optical_Flow_Frame_" + "%05d"%(index) +'.flo'

        image_RGB = cv2.imread(image_path)
        flow_data_gt = read_flow(flow_file_gt)

        inference_cylinder_RGB_file = '/home/yiheng/project/test_data/city100R_C/image/Frame_' + "%05d"%(index) +'.png'
        inference_cylinder_RGB = cv2.imread(inference_cylinder_RGB_file)
        test_cylinder_RGB = libexample.RGB_equirect_to_cylinder(image_RGB, image_RGB.shape[0], image_RGB.shape[1], image_RGB.shape[2])
        print(calculate_MSE(inference_cylinder_RGB, test_cylinder_RGB))

        inference_cylinder_flow_file = "/home/yiheng/project/test_data/city100R_C/flow/original_file_Optical_Flow_Frame_" + "%05d"%(index) +'.flo'
        inference_cylinder_flow = read_flow(inference_cylinder_flow_file)
        test_cylinder_flow = libexample.flow_equirect_to_cylinder(flow_data_gt.astype(np.float32), flow_data_gt.shape[0], flow_data_gt.shape[1], flow_data_gt.shape[2])
        test_cylinder_flow = test_cylinder_flow.astype(np.float32)
        print(calculate_EPE(inference_cylinder_flow, test_cylinder_flow))

        inference_cubepadding_RGB_file = '/home/yiheng/project/test_data/city100R_P/image/Frame_' + "%05d"%(index) +'.png'
        inference_cubepadding_RGB = cv2.imread(inference_cubepadding_RGB_file)
        test_cubepadding_RGB = libexample.RGB_equirect_to_cubepadding(image_RGB, image_RGB.shape[0], image_RGB.shape[1], image_RGB.shape[2])
        print(calculate_MSE(inference_cubepadding_RGB, test_cubepadding_RGB))

        inference_cubepadding_flow_file = "/home/yiheng/project/test_data/city100R_P/flow/original_file_Optical_Flow_Frame_" + "%05d"%(index) +'.flo'
        inference_cubepadding_flow = read_flow(inference_cubepadding_flow_file)
        test_cubepadding_flow = libexample.flow_equirect_to_cubepadding(flow_data_gt.astype(np.float32), flow_data_gt.shape[0], flow_data_gt.shape[1], flow_data_gt.shape[2])
        test_cubepadding_flow = test_cubepadding_flow.astype(np.float32)
        print(calculate_EPE(inference_cubepadding_flow, test_cubepadding_flow))

        test_cylinder_to_equirect_flow = libexample.flow_cylinder_to_equirect(inference_cylinder_flow.astype(np.float32), inference_cylinder_flow.shape[0], inference_cylinder_flow.shape[1], inference_cylinder_flow.shape[2])
        test_cylinder_to_equirect_flow = test_cylinder_to_equirect_flow.astype(np.float32)
        print(calculate_EPE(flow_data_gt, test_cylinder_to_equirect_flow))

        test_cubepadding_to_equirect_flow = flow_cubepadding_to_equirect(inference_cubepadding_flow)
        print(calculate_EPE(flow_data_gt, test_cubepadding_to_equirect_flow))



