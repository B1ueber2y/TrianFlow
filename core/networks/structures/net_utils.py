import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def warp_flow(x, flow, use_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    Inputs:
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow

    Returns:
    ouptut: [B, C, H, W]
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if grid.shape != flow.shape:
        raise ValueError('the shape of grid {0} is not equal to the shape of flow {1}.'.format(grid.shape, flow.shape))
    if x.is_cuda:
        grid = grid.to(x.get_device())
    vgrid = grid + flow

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    if use_mask:
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.get_device())
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask
    else:
        return output

if __name__ == '__main__':
    x = np.ones([1,1,10,10])
    flow = np.stack([np.ones([1,10,10])*3.0, np.zeros([1,10,10])], axis=1)
    y = warp_flow(torch.from_numpy(x).cuda().float(),torch.from_numpy(flow).cuda().float()).cpu().detach().numpy()
    print(y)

