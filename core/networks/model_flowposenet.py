import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from structures import *
from pytorch_ssim import SSIM
from model_flow import Model_flow
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'visualize'))
from visualizer import *
from profiler import Profiler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import cv2

def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()
    return mean_value


def edge_aware_smoothness_loss(pred_disp, img, max_scales):
    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def get_edge_smoothness(img, pred):
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x),
                                          1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y),
                                          1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    s = 0
    for scaled_disp in pred_disp:
        s += 1
        if s > max_scales:
            break

        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp) * weight
        weight /= 4.0

    return loss


def compute_smooth_loss(tgt_depth, tgt_img, ref_depth, ref_img, max_scales=1):
    loss = edge_aware_smoothness_loss(tgt_depth, tgt_img, max_scales)
    loss = edge_aware_smoothness_loss(ref_depth, ref_img, max_scales)

    return loss


class Model_flowposenet(nn.Module):
    def __init__(self, cfg):
        super(Model_flowposenet, self).__init__()
        assert cfg.depth_scale == 1
        self.pose_net = FlowPoseNet()
        self.model_flow = Model_flow(cfg)
        self.depth_net = Depth_Model(cfg.depth_scale)
    
    def compute_pairwise_loss(self, tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic):
        ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth,
                                                                                    pose, intrinsic, 'zeros')

        diff_img = (tgt_img - ref_img_warped).abs()

        diff_depth = ((computed_depth - projected_depth).abs() /
                    (computed_depth + projected_depth).abs()).clamp(0, 1)

        
        ssim_map = (0.5*(1 - SSIM(tgt_img, ref_img_warped))).clamp(0, 1)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

        # Modified in 01.19.2020
        #weight_mask = (1 - diff_depth)
        #diff_img = diff_img * weight_mask
        

        # compute loss
        reconstruction_loss = diff_img.mean()
        geometry_consistency_loss = diff_depth.mean()
        #reconstruction_loss = mean_on_mask(diff_img, valid_mask)
        #geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

        return reconstruction_loss, geometry_consistency_loss

    
    def disp2depth(self, disp, min_depth=0.01, max_depth=80.0):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def infer_depth(self, img):
        b, img_h, img_w = img.shape[0], img.shape[2], img.shape[3]
        disp_list = self.depth_net(img)
        disp, depth = self.disp2depth(disp_list[0])
        return disp_list[0]
    
    def inference(self, img1, img2, K, K_inv):
        flow = self.model_flow.inference_flow(img1, img2)
        return flow, None, None, None, None, None
    
    def inference_flow(self, img1, img2):
        flow = self.model_flow.inference_flow(img1, img2)
        return flow
    
    def infer_pose(self, img1, img2, K, K_inv):
        img_h, img_w = img1.shape[2], img1.shape[3]
        flow = self.model_flow.inference_flow(img1, img2)
        flow[:,0,:,:] /= img_w
        flow[:,1,:,:] /= img_h
        pose = self.pose_net(flow)
        return pose

    def forward(self, inputs):
        # initialization
        images, K_ms, K_inv_ms = inputs
        K, K_inv = K_ms[:,0,:,:], K_inv_ms[:,0,:,:]
        assert (images.shape[1] == 3)
        img_h, img_w = int(images.shape[2] / 2), images.shape[3] 
        img1, img2 = images[:,:,:img_h,:], images[:,:,img_h:,:]
        b = img1.shape[0]
        visualizer = Visualizer_debug('./vis/', img1=255*img1.permute([0,2,3,1]).detach().cpu().numpy(), \
            img2=255*img2.permute([0,2,3,1]).detach().cpu().numpy())
        
        # Flow Network
        loss_pack, fwd_flow, bwd_flow, img1_valid_mask, img2_valid_mask, img1_flow_diff_mask, img2_flow_diff_mask = self.model_flow(inputs, output_flow=True, use_flow_loss=False)
        fwd_flow[:,0,:,:] /= img_w
        fwd_flow[:,1,:,:] /= img_h
        bwd_flow[:,0,:,:] /= img_w
        bwd_flow[:,1,:,:] /= img_h

        # Pose Network
        pose = self.pose_net(fwd_flow)
        pose_inv = self.pose_net(bwd_flow)
        disp1_list = self.depth_net(img1) # Nscales * [B, 1, H, W]
        disp2_list = self.depth_net(img2)
        disp1, depth1 = self.disp2depth(disp1_list[0])
        disp2, depth2 = self.disp2depth(disp2_list[0])
        #pdb.set_trace()

        loss_1, loss_3 = self.compute_pairwise_loss(img1, img2, depth1, depth2, pose, K)
        loss_1_2, loss_3_2 = self.compute_pairwise_loss(img2, img1, depth2, depth1, pose_inv, K)
        loss_ph = loss_1 + loss_1_2
        loss_pj = loss_3 + loss_3_2

        loss_2 = compute_smooth_loss([depth1], img1, [depth2], img2)

        loss_pack['pt_depth_loss'] = torch.zeros([2]).to(loss_2.get_device()).requires_grad_()
        loss_pack['pj_depth_loss'], loss_pack['flow_error'] = loss_pj, loss_ph
        loss_pack['depth_smooth_loss'] = loss_2
        #loss_pack['depth_smooth_loss'] = torch.zeros([2]).to(loss_2.get_device()).requires_grad_()
        loss_pack['geo_loss'] = torch.zeros([2]).to(loss_2.get_device()).requires_grad_()
        return loss_pack


