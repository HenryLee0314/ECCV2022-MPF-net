import torch
from torch import nn

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow, p=2, dim=1).mean()


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

        # self.equirect_loss_weight = EquirectLossWeight(512, 1024)
        # self.cubemap_loss_weight = CubemapLossWeight(1024, 1024)
        # self.cylinder_loss_weight = CylinderWeight("/home/liyihe/Desktop/cylinder_weight_map.weight")


    def forward(self, output, target):
        lossvalue = torch.norm(output-target, p=2, dim=1).mean()
        # lossvalue = self.cylinder_loss_weight(lossvalue).mean()
        return lossvalue


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]