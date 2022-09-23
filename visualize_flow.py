import cv2
import math
import numpy as np
import os

TAG_FLOAT = 202021.25  # Do NOT edit this TAG as it is the flow file header

vis_scale = 100 # can change this number for a better visualization, 100 or 200 or 300 or as you like


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


def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    maxrad = vis_scale#np.sqrt(2048 ** 2 + 1024 ** 2)

    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


# return: the error map, mean error, and max value of the map
def calculate_EPE(flow_gt, flow_target):
    x = flow_target[:,:,0] - flow_gt[:,:,0]
    y = flow_target[:,:,1] - flow_gt[:,:,1]
    d = np.sqrt(x ** 2 + y ** 2)
    return d, (np.sum(d) / (512*1024)), np.max(d)



DISK_PATH = "/media/yiheng/T2/"

if __name__ == '__main__':
    for index in range(138):
        # !!!important information!!!
        # Skip index[0] because our GT flow is from current frame to previous frame (from index[1] to index[0]).
        # In other words, index[0] doesn't have its target.
        # Similar, for the RGB, use the index[1] frame because it is the source RGB image.

        # However, in terms of the flow we predicted, its index represent the image pair (begin from 0)
        # In other words, the flow GT_flow[1] should be compaired with predicted_flow[0], and inference RGB[1]
        if index == 0:
            continue
#########################
# prepare file input path
#########################
        # City100R
        # image_src_path = DISK_PATH + '__DATASET__/City_100_r/image/Frame_' + "%05d"%(index) +'.png'
        # image_tar_path = DISK_PATH + '__DATASET__/City_100_r/image/Frame_' + "%05d"%(index-1) +'.png'
        # flow_file_gt = DISK_PATH + "__DATASET__/City_100_r/flow/original_file_Optical_Flow_Frame_" + "%05d"%(index) +'.flo'
        # flow_e_c = DISK_PATH + "__ECCV__DONE__/correct_fusion_for_city/tmp/" + "%06d"%(index-1) +'.flo'
        # flow_e_p = DISK_PATH + "2022_Feb_26_models/City100R_fusion_output_E_P/tmp/" + "%06d"%(index-1) +'.flo'
        # flow_c_p = DISK_PATH + "2022_Feb_28_C_P_models/tmp_City100R_predict_fusion_CP/tmp/" + "%06d"%(index-1) +'.flo'
        # flow_e = DISK_PATH + "__TEMP__/dataset_for_fusion/city/testing/e/" + "%06d"%(index-1) +'.flo'
        # flow_c = DISK_PATH + "__TEMP__/dataset_for_fusion/city/testing/c/" + "%06d"%(index-1) +'.flo'
        # flow_p = DISK_PATH + "__ECCV__/___DATASET___/__FUSION__/City100R_P/" + "%06d"%(index-1) +'.flo'

        # EFT100
        image_src_path = DISK_PATH + '__DATASET__/EFTs_Car100/image/Frame_' + "%05d"%(index) +'.png'
        image_tar_path = DISK_PATH + '__DATASET__/EFTs_Car100/image/Frame_' + "%05d"%(index-1) +'.png'
        flow_file_gt = DISK_PATH + "__DATASET__/EFTs_Car100/flow/original_file_Optical_Flow_Frame_" + "%05d"%(index) +'.flo'
        flow_e_c = DISK_PATH + "__ECCV__DONE__/correct_fusion_for_EFT/tmp/" + "%06d"%(index-1) +'.flo'
        flow_e_p = DISK_PATH + "2022_Feb_26_models/EFT100_fusion_output_E_P/tmp/" + "%06d"%(index-1) +'.flo'
        flow_c_p = DISK_PATH + "2022_Feb_28_C_P_models/tmp_EFT_predict_fusion_CP/tmp/" + "%06d"%(index-1) +'.flo'
        flow_e = DISK_PATH + "__TEMP__/dataset_for_fusion/EFT/testing/e/" + "%06d"%(index-1) +'.flo'
        flow_c = DISK_PATH + "__TEMP__/dataset_for_fusion/EFT/testing/c/" + "%06d"%(index-1) +'.flo'
        flow_p = DISK_PATH + "__ECCV__/___DATASET___/__FUSION__/EFTCar100_P/" + "%06d"%(index-1) +'.flo'

#########################
# read input files
#########################
        image_src_RGB = cv2.imread(image_src_path)
        image_tar_RGB = cv2.imread(image_tar_path)
        flow_gt = read_flow(flow_file_gt)
        flow_e_c = read_flow(flow_e_c)
        flow_e_p = read_flow(flow_e_p)
        flow_c_p = read_flow(flow_c_p)
        flow_e = read_flow(flow_e)
        flow_c = read_flow(flow_c)
        flow_p = read_flow(flow_p)


#########################
# calculate EPE error map and its mean & max
#########################

        flow_e_c_EPE_map, flow_e_c_mean_EPE, flow_e_c_max_EPE = calculate_EPE(flow_gt, flow_e_c)
        flow_e_p_EPE_map, flow_e_p_mean_EPE, flow_e_p_max_EPE = calculate_EPE(flow_gt, flow_e_p)
        flow_c_p_EPE_map, flow_c_p_mean_EPE, flow_c_p_max_EPE = calculate_EPE(flow_gt, flow_c_p)
        flow_e_EPE_map, flow_e_mean_EPE, flow_e_max_EPE = calculate_EPE(flow_gt, flow_e)
        flow_c_EPE_map, flow_c_mean_EPE, flow_c_max_EPE = calculate_EPE(flow_gt, flow_c)
        flow_p_EPE_map, flow_p_mean_EPE, flow_p_max_EPE = calculate_EPE(flow_gt, flow_p)

#########################
# adjust EPE map for a better visualization
#########################

        max_EPE = max(flow_e_max_EPE, flow_c_max_EPE, flow_p_max_EPE, flow_e_c_max_EPE, flow_e_p_max_EPE, flow_c_p_max_EPE)

        flow_e_c_EPE_map = (((flow_e_c_EPE_map / max_EPE)** 0.25 )*255).astype(np.uint8)
        flow_e_p_EPE_map = (((flow_e_p_EPE_map / max_EPE)** 0.25 )*255).astype(np.uint8)
        flow_c_p_EPE_map = (((flow_c_p_EPE_map / max_EPE)** 0.25 )*255).astype(np.uint8)
        flow_e_EPE_map = (((flow_e_EPE_map / max_EPE)** 0.25 )*255).astype(np.uint8)
        flow_c_EPE_map = (((flow_c_EPE_map / max_EPE)** 0.25 )*255).astype(np.uint8)
        flow_p_EPE_map = (((flow_p_EPE_map / max_EPE)** 0.25 )*255).astype(np.uint8)

#########################
# flow to RGB and then do the RGB2BGR (opencv things)
#########################
        flow_gt = cv2.cvtColor(flow2img(flow_gt), cv2.COLOR_RGB2BGR)
        flow_e_c = cv2.cvtColor(flow2img(flow_e_c), cv2.COLOR_RGB2BGR)
        flow_e_p = cv2.cvtColor(flow2img(flow_e_p), cv2.COLOR_RGB2BGR)
        flow_c_p = cv2.cvtColor(flow2img(flow_c_p), cv2.COLOR_RGB2BGR)
        flow_e = cv2.cvtColor(flow2img(flow_e), cv2.COLOR_RGB2BGR)
        flow_c = cv2.cvtColor(flow2img(flow_c), cv2.COLOR_RGB2BGR)
        flow_p = cv2.cvtColor(flow2img(flow_p), cv2.COLOR_RGB2BGR)
    

#########################
# EPE gray map to BGR
#########################
        flow_e_c_EPE_map = cv2.cvtColor(flow_e_c_EPE_map, cv2.COLOR_GRAY2BGR)
        flow_e_p_EPE_map = cv2.cvtColor(flow_e_p_EPE_map, cv2.COLOR_GRAY2BGR)
        flow_c_p_EPE_map = cv2.cvtColor(flow_c_p_EPE_map, cv2.COLOR_GRAY2BGR)
        flow_e_EPE_map = cv2.cvtColor(flow_e_EPE_map, cv2.COLOR_GRAY2BGR)
        flow_c_EPE_map = cv2.cvtColor(flow_c_EPE_map, cv2.COLOR_GRAY2BGR)
        flow_p_EPE_map = cv2.cvtColor(flow_p_EPE_map, cv2.COLOR_GRAY2BGR)

#########################
# add text label
#########################
        # below 4 line is currently no use, but if you want to add EPE, it is good for inference. 
        # cv2.putText(flow_e_c, "(Fusion) EPE:%.3f"%flow_e_c_mean_EPE, (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        # cv2.putText(flow_e, "(E) EPE:%.3f"%flow_e_mean_EPE, (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        # cv2.putText(flow_c, "(C) EPE:%.3f"%flow_c_mean_EPE, (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        # cv2.putText(flow_p, "(P) EPE:%.3f"%flow_p_mean_EPE, (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)

        cv2.putText(image_src_RGB, "source", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        cv2.putText(image_tar_RGB, "target", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        cv2.putText(flow_gt, "GT flow", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)


        cv2.putText(flow_e, "(E) Flow Prediction", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        cv2.putText(flow_c, "(C) Flow Prediction", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        cv2.putText(flow_p, "(P) Flow Prediction", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)

        cv2.putText(flow_e_EPE_map, "(E) EPE Heat Map", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(flow_c_EPE_map, "(C) EPE Heat Map", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(flow_p_EPE_map, "(P) EPE Heat Map", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(flow_e_c, "E-C Fusion", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        cv2.putText(flow_e_p, "E-P Fusion", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
        cv2.putText(flow_c_p, "C-P Fusion", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)

        cv2.putText(flow_e_c_EPE_map, "E-C Fusion EPE Heat Map", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(flow_e_p_EPE_map, "E-P Fusion EPE Heat Map", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(flow_c_p_EPE_map, "C-P Fusion EPE Heat Map", (10, 512-475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)




#########################
# make to a single image
#########################

        image_col_gt = np.vstack((image_src_RGB, image_tar_RGB, flow_gt, flow_gt))
        image_col_e = np.vstack((flow_e, flow_e_EPE_map, flow_e_c, flow_e_c_EPE_map))
        image_col_c = np.vstack((flow_c, flow_c_EPE_map, flow_e_p, flow_e_p_EPE_map))
        image_col_p = np.vstack((flow_p, flow_p_EPE_map, flow_c_p, flow_c_p_EPE_map))

        out = np.hstack((image_col_gt, image_col_e, image_col_c, image_col_p))

        resized = cv2.resize(out, (512*4, 256*4), interpolation = cv2.INTER_AREA)
        cv2.imshow('image', resized.astype(np.uint8)) 
        # cv2.imwrite('image_' + str(index) + '.png', resized.astype(np.uint8)) 


        cv2.waitKey(0)

    cv2.destroyAllWindows()