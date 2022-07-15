import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable




class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def convtranspose3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()
        self.relu = self.mish
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResEncoder(nn.Module):
    def __init__(self, in_features, out_features, stride=1, downsample=None):
        super(ResEncoder, self).__init__()
        self.conv1 = conv3x3(in_features, out_features, stride)
        self.bn1 = nn.BatchNorm2d(out_features)
        #self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()
        self.relu = self.mish
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_features),
            )
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResDecoder(nn.Module):
    def __init__(self, in_features, out_features, stride=1, upsample=None):
        super(ResDecoder, self).__init__()
        self.conv1 = convtranspose3x3(in_features, out_features, stride)
        self.bn1 = nn.BatchNorm2d(out_features)
        # self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()
        self.relu = self.mish
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_features, out_features, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False),
                nn.BatchNorm2d(out_features),
            )
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.upsample is not None:
            residual = self.upsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class FusionNet(torch.nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        feature = 32

        self.encoder_1 = nn.Sequential(nn.Conv2d(9, feature, 3, stride=1, padding=1), Mish())

        self.encoder_2 = nn.Sequential(ResEncoder(feature, 2 * feature, 2), Mish())
        self.encoder_3 = nn.Sequential(ResEncoder(2 * feature, 4 * feature, 2), Mish())
        self.encoder_4 = nn.Sequential(ResEncoder(4 * feature, 8 * feature, 2), Mish())

        self.conn_1 = nn.Sequential(nn.Conv2d(8 * feature, 8 * feature, 1, stride=1, padding=0), Mish())
        self.conn_2 = nn.Sequential(nn.Conv2d(8 * feature, 8 * feature, 1, stride=1, padding=0), Mish())
        self.conn_3 = nn.Sequential(nn.Conv2d(8 * feature, 8 * feature, 1, stride=1, padding=0), Mish())

        self.decoder_4 = nn.Sequential(ResDecoder(8 * feature * 2, 4 * feature, 2), Mish())
        self.decoder_3 = nn.Sequential(ResDecoder(4 * feature * 2, 2 * feature, 2), Mish())
        self.decoder_2 = nn.Sequential(ResDecoder(2 * feature * 2, feature, 2), Mish())
        self.decoder_1 = nn.Sequential(nn.ConvTranspose2d(feature * 2, 1, 3, stride=1, padding=1, output_padding=0))

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        vgrid[:,0,:,:] = torch.where(vgrid[:,0] > W, vgrid[:,0] - W, vgrid[:,0])
        vgrid[:,0,:,:] = torch.where(vgrid[:,0] < 0, W + vgrid[:,0], vgrid[:,0])

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)


        mask[mask<0.999] = 0
        mask[mask>0] = 1
        
        return output*mask
  

    def forward(self, input_e, input_c, input_i):
        # print("===========")
        # print(input_e.shape)
        # print(input_c.shape)
        # print(input_i.shape)
        # print("===========")
        

        warp_e = self.warp(input_i, input_e)
        warp_c = self.warp(input_i, input_c)

        input = torch.cat((warp_e, warp_c, input_i), 1)

        # print("input", input.shape)
        # print(warp_c.shape)
        # image_e = warp_e[0].cpu().data.numpy()
        # image_e = np.swapaxes(np.swapaxes(image_e, 0, 1), 1, 2)
        # cv2.imwrite("image_e.png", image_e)



        # print("input.shape", input.shape)

        encoder_1_out = self.encoder_1(input)
        # print("encoder_1_out", encoder_1_out.shape)

        encoder_2_out = self.encoder_2(encoder_1_out)
        # print("encoder_2_out", encoder_2_out.shape)

        encoder_3_out = self.encoder_3(encoder_2_out)
        # print("encoder_3_out", encoder_3_out.shape)

        encoder_4_out = self.encoder_4(encoder_3_out)
        # print("encoder_4_out", encoder_4_out.shape)

        out = self.conn_1(encoder_4_out)
        # print("conn_1", out.shape)
        out = self.conn_2(out)
        # print("conn_2", out.shape)
        out = self.conn_3(out)
        # print("conn_3", out.shape)

        out = torch.cat((out, encoder_4_out), 1)
        decoder_4_out = self.decoder_4(out)
        # print("decoder_4_out", decoder_4_out.shape)

        out = torch.cat((decoder_4_out, encoder_3_out), 1)
        decoder_3_out = self.decoder_3(out)
        # print("decoder_3_out", decoder_3_out.shape)

        out = torch.cat((decoder_3_out, encoder_2_out), 1)
        decoder_2_out = self.decoder_2(out)
        # print("decoder_2_out", decoder_2_out.shape)

        out = torch.cat((decoder_2_out, encoder_1_out), 1)
        t = self.decoder_1(out)
        # print("t", t.shape)

        out = input_e * t + (1-t) * input_c

        # print("final out", out.shape)

        return out

def get_fusion_net(path=None):
    model = FusionNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'], strict=False)
        else:
            model.load_state_dict(data)
    return model