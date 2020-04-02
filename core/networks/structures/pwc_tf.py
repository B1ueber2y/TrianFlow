import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from net_utils import conv, deconv, warp_flow
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'external'))
# from correlation_package.correlation import Correlation
# from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
import torch.nn.functional as F
#from spatial_correlation_sampler import spatial_correlation_sample

class PWC_tf(nn.Module):
    def __init__(self, md=4):
        super(PWC_tf, self).__init__()
        self.corr = self.corr_naive
        # self.corr = self.correlate
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        #dd = np.cumsum([128,128,96,64,32])
        dd = np.array([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv6_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = self.predict_flow(dd[3]+dd[4])
        #self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        #self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+2
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv5_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = self.predict_flow(dd[3]+dd[4]) 
        #self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        #self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+2
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv4_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = self.predict_flow(dd[3]+dd[4]) 
        #self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        #self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+2
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv3_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = self.predict_flow(dd[3]+dd[4])
        #self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        #self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+2
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv2_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = self.predict_flow(dd[3]+dd[4]) 
        #self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(dd[4]+2,  128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = self.predict_flow(32)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_zeros_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        '''
    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

    def warp(self, x, flow):
        return warp_flow(x, flow, use_mask=False)

    def corr_naive(self, input1, input2, d=4):
        # naive pytorch implementation of the correlation layer.
        assert (input1.shape == input2.shape)
        batch_size, feature_num, H, W = input1.shape[0:4]
        input2 = F.pad(input2, (d,d,d,d), value=0)
        cv = []
        for i in range(2 * d + 1):
            for j in range(2 * d + 1):
                cv.append((input1 * input2[:, :, i:(i + H), j:(j + W)]).mean(1).unsqueeze(1))
        return torch.cat(cv, 1)
    
    def forward(self, feature_list_1, feature_list_2, img_hw):
        c11, c12, c13, c14, c15, c16 = feature_list_1
        c21, c22, c23, c24, c25, c26 = feature_list_2

        corr6 = self.corr(c16, c26) 
        x0 = self.conv6_0(corr6)
        x1 = self.conv6_1(x0)
        x2 = self.conv6_2(torch.cat((x0,x1),1))
        x3 = self.conv6_3(torch.cat((x1,x2),1))
        x4 = self.conv6_4(torch.cat((x2,x3),1))
        flow6 = self.predict_flow6(torch.cat((x3,x4),1))
        up_flow6 = F.interpolate(flow6, scale_factor=2.0, mode='bilinear')*2.0

        warp5 = self.warp(c25, up_flow6)
        corr5 = self.corr(c15, warp5) 
        x = torch.cat((corr5, c15, up_flow6), 1)
        x0 = self.conv5_0(x)
        x1 = self.conv5_1(x0)
        x2 = self.conv5_2(torch.cat((x0,x1),1))
        x3 = self.conv5_3(torch.cat((x1,x2),1))
        x4 = self.conv5_4(torch.cat((x2,x3),1))
        flow5 = self.predict_flow5(torch.cat((x3,x4),1))
        flow5 = flow5 + up_flow6
        up_flow5 = F.interpolate(flow5, scale_factor=2.0, mode='bilinear')*2.0

       
        warp4 = self.warp(c24, up_flow5)
        corr4 = self.corr(c14, warp4)  
        x = torch.cat((corr4, c14, up_flow5), 1)
        x0 = self.conv4_0(x)
        x1 = self.conv4_1(x0)
        x2 = self.conv4_2(torch.cat((x0,x1),1))
        x3 = self.conv4_3(torch.cat((x1,x2),1))
        x4 = self.conv4_4(torch.cat((x2,x3),1))
        flow4 = self.predict_flow4(torch.cat((x3,x4),1))
        flow4 = flow4 + up_flow5
        up_flow4 = F.interpolate(flow4, scale_factor=2.0, mode='bilinear')*2.0

        warp3 = self.warp(c23, up_flow4)
        corr3 = self.corr(c13, warp3) 
        x = torch.cat((corr3, c13, up_flow4), 1)
        x0 = self.conv3_0(x)
        x1 = self.conv3_1(x0)
        x2 = self.conv3_2(torch.cat((x0,x1),1))
        x3 = self.conv3_3(torch.cat((x1,x2),1))
        x4 = self.conv3_4(torch.cat((x2,x3),1))
        flow3 = self.predict_flow3(torch.cat((x3,x4),1))
        flow3 = flow3 + up_flow4
        up_flow3 = F.interpolate(flow3, scale_factor=2.0, mode='bilinear')*2.0


        warp2 = self.warp(c22, up_flow3) 
        corr2 = self.corr(c12, warp2)
        x = torch.cat((corr2, c12, up_flow3), 1)
        x0 = self.conv2_0(x)
        x1 = self.conv2_1(x0)
        x2 = self.conv2_2(torch.cat((x0,x1),1))
        x3 = self.conv2_3(torch.cat((x1,x2),1))
        x4 = self.conv2_4(torch.cat((x2,x3),1))
        flow2 = self.predict_flow2(torch.cat((x3,x4),1))
        flow2 = flow2 + up_flow3
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(torch.cat([flow2, x4], 1)))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        img_h, img_w = img_hw[0], img_hw[1]
        flow2 = F.interpolate(flow2 * 4.0, [img_h, img_w], mode='bilinear')
        flow3 = F.interpolate(flow3 * 4.0, [img_h // 2, img_w // 2], mode='bilinear')
        flow4 = F.interpolate(flow4 * 4.0, [img_h // 4, img_w // 4], mode='bilinear')
        flow5 = F.interpolate(flow5 * 4.0, [img_h // 8, img_w // 8], mode='bilinear')
        
        return [flow2, flow3, flow4, flow5]

