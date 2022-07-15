import numpy as np
import math
import torch
import cv2

def generate_weight(H, W):
    weight = np.zeros([1, H, W])
    for h in range(H):
        for w in range (W):
            phi = abs(float(h)/H-0.5) * math.pi
            weight[0,h,w] = math.cos(phi)
    return weight

def generate_cubemap_weight(H, W):
    weight = np.zeros([1, H, W])
    for h in range(H):
        for w in range (W):
            weight[0,h,w] = 0
            if H / 4 <= h and h < H / 2 and W / 2 <= w and w < W / 4 * 3:
                weight[0,h,w] = 255
            if H / 2 <= h and h < H / 4 * 3:
                weight[0,h,w] = 255;
            if H / 4 * 3 <= h and h < H and W / 2 <= w and w < W / 4 * 3:
                weight[0,h,w] = 255;
    return weight


class CubemapLossWeight(nn.Module):
    def __init__(self, H, W):
        super(CubemapLossWeight, self).__init__()
        weight = generate_cubemap_weight(H, W)
        self.weight = torch.from_numpy(weight)

    def forward(self, loss):
        return loss * self.weight.cuda()

class EquirectLossWeight(nn.Module):
    def __init__(self, H, W):
        super(EquirectLossWeight, self).__init__()
        weight = generate_weight(H, W)
        self.weight = torch.from_numpy(weight)

    def forward(self, loss):
        return loss * self.weight.cuda()

def read_weight(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    f = open(file,'rb')
    # flo_number = np.fromfile(f, np.float32, count=1)[0]
    w = np.fromfile(f, np.int32, count=1)
    # print("w: ", w)
    h = np.fromfile(f, np.int32, count=1)
    # print("h: ", h)
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # data = np.fromfile(f, np.float32, count=2*w*h)
    data = np.fromfile(f, np.float32, count=w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    weight = np.resize(data, (int(h), int(w)))    
    f.close()

    return weight, h, w

class CylinderWeight(nn.Module):
    def __init__(self, file):
        super(CylinderWeight, self).__init__()
        weight, h, w = read_weight(file)
        weight = weight / np.amax(weight)
        weight = cv2.resize(weight, (int(w), int(3/4*w)), interpolation = cv2.INTER_AREA)
        self.weight = torch.from_numpy(weight)

    def forward(self, loss):
        return loss * self.weight.cuda()

