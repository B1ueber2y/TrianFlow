import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from net_utils import conv
import torch
import torch.nn as nn

class FeaturePyramid(nn.Module):
    def __init__(self):
        super(FeaturePyramid, self).__init__()
        self.conv1 = conv(3,   16, kernel_size=3, stride=2)
        self.conv2 = conv(16,  16, kernel_size=3, stride=1)
        self.conv3 = conv(16,  32, kernel_size=3, stride=2)
        self.conv4 = conv(32,  32, kernel_size=3, stride=1)
        self.conv5 = conv(32,  64, kernel_size=3, stride=2)
        self.conv6 = conv(64,  64, kernel_size=3, stride=1)
        self.conv7 = conv(64,  96, kernel_size=3, stride=2)
        self.conv8 = conv(96,  96, kernel_size=3, stride=1)
        self.conv9 = conv(96, 128, kernel_size=3, stride=2)
        self.conv10 = conv(128, 128, kernel_size=3, stride=1)
        self.conv11 = conv(128, 196, kernel_size=3, stride=2)
        self.conv12 = conv(196, 196, kernel_size=3, stride=1)
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.constant_(m.weight.data, 0.0)
                if m.bias is not None:
                    m.bias.data.zero_()
        '''
    def forward(self, img):
        cnv2 = self.conv2(self.conv1(img))
        cnv4 = self.conv4(self.conv3(cnv2))
        cnv6 = self.conv6(self.conv5(cnv4))
        cnv8 = self.conv8(self.conv7(cnv6))
        cnv10 = self.conv10(self.conv9(cnv8))
        cnv12 = self.conv12(self.conv11(cnv10))
        return cnv2, cnv4, cnv6, cnv8, cnv10, cnv12

