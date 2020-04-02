import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import numpy as np
from structures import *
from model_flow import Model_flow
import pdb
import cv2

class Model_triangulate_pose(nn.Module):
    def __init__(self, cfg):
        super(Model_triangulate_pose, self).__init__()
        self.model_flow = Model_flow(cfg)
        self.mode = cfg.mode
        if cfg.dataset == 'nyuv2':
            self.inlier_thres = 0.1
            self.rigid_thres = 1.0
        else:
            self.inlier_thres = 0.1
            self.rigid_thres = 0.5
        self.filter = reduced_ransac(check_num=cfg.ransac_points, thres=self.inlier_thres, dataset=cfg.dataset)
    
    def meshgrid(self, h, w):
        xx, yy = np.meshgrid(np.arange(0,w), np.arange(0,h))
        meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
        meshgrid = torch.from_numpy(meshgrid)
        return meshgrid
    
    def compute_epipolar_loss(self, fmat, match, mask):
        # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
        num_batch = match.shape[0]
        match_num = match.shape[-1]

        points1 = match[:,:2,:]
        points2 = match[:,2:,:]
        ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
        points1 = torch.cat([points1, ones], 1) # [b,3,n]
        points2 = torch.cat([points2, ones], 1).transpose(1,2) # [b,n,3]

        # compute fundamental matrix loss
        fmat = fmat.unsqueeze(1)
        fmat_tiles = fmat.view([-1,3,3])
        epi_lines = fmat_tiles.bmm(points1) #[b,3,n]  [b*n, 3, 1]
        dist_p2l = torch.abs((epi_lines.permute([0, 2, 1]) * points2).sum(-1, keepdim=True)) # [b,n,1]
        a = epi_lines[:,0,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        b = epi_lines[:,1,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        dist_div = torch.sqrt(a*a + b*b) + 1e-6
        dist_map = dist_p2l / dist_div # [B, n, 1]
        loss = (dist_map * mask.transpose(1,2)).mean([1,2]) / mask.mean([1,2])
        return loss, dist_map

    def get_rigid_mask(self, dist_map):
        rigid_mask = (dist_map < self.rigid_thres).float()
        inlier_mask = (dist_map < self.inlier_thres).float()
        rigid_score = rigid_mask * 1.0 / (1.0 + dist_map)
        return rigid_mask, inlier_mask, rigid_score
    
    def inference(self, img1, img2, K, K_inv):
        batch_size, img_h, img_w = img1.shape[0], img1.shape[2], img1.shape[3]
        
        fwd_flow, bwd_flow, img1_valid_mask, img2_valid_mask, img1_flow_diff_mask, img2_flow_diff_mask = self.model_flow.inference_corres(img1, img2)
        
        grid = self.meshgrid(img_h, img_w).float().to(img1.get_device()).unsqueeze(0).repeat(batch_size,1,1,1) #[b,2,h,w]
        corres = torch.cat([(grid[:,0,:,:] + fwd_flow[:,0,:,:]).clamp(0,img_w-1.0).unsqueeze(1), \
            (grid[:,1,:,:] + fwd_flow[:,1,:,:]).clamp(0,img_h-1.0).unsqueeze(1)], 1)
        match = torch.cat([grid, corres], 1) # [b,4,h,w]

        img1_score_mask = img1_valid_mask * 1.0 / (0.1 + img1_flow_diff_mask.mean(1).unsqueeze(1))
        F_final = self.filter(match, img1_score_mask)
        geo_loss, rigid_mask = self.compute_epipolar_loss(F_final, match.view([batch_size,4,-1]), img1_valid_mask.view([batch_size,1,-1]))
        img1_rigid_mask = (rigid_mask.view([batch_size,img_h,img_w,1]) < self.inlier_thres).float()
        
        return F_final, img1_valid_mask, img1_rigid_mask.permute(0,3,1,2), fwd_flow, match

    def forward(self, inputs, output_F=False, visualizer=None):
        images, K_ms, K_inv_ms = inputs
        K, K_inv = K_ms[:,0,:,:], K_inv_ms[:,0,:,:]
        assert (images.shape[1] == 3)
        img_h, img_w = int(images.shape[2] / 2), images.shape[3] 
        img1, img2 = images[:,:,:img_h,:], images[:,:,img_h:,:]
        batch_size = img1.shape[0]

        if self.mode == 'depth':
            loss_pack, fwd_flow, bwd_flow, img1_valid_mask, img2_valid_mask, img1_flow_diff_mask, img2_flow_diff_mask = self.model_flow(inputs, output_flow=True, use_flow_loss=False)
        else:
            loss_pack, fwd_flow, bwd_flow, img1_valid_mask, img2_valid_mask, img1_flow_diff_mask, img2_flow_diff_mask = self.model_flow(inputs, output_flow=True)
        
        grid = self.meshgrid(img_h, img_w).float().to(img1.get_device()).unsqueeze(0).repeat(batch_size,1,1,1) #[b,2,h,w]
        
        fwd_corres = torch.cat([(grid[:,0,:,:] + fwd_flow[:,0,:,:]).unsqueeze(1), (grid[:,1,:,:] + fwd_flow[:,1,:,:]).unsqueeze(1)], 1)
        fwd_match = torch.cat([grid, fwd_corres], 1) # [b,4,h,w]

        bwd_corres = torch.cat([(grid[:,0,:,:] + bwd_flow[:,0,:,:]).unsqueeze(1), (grid[:,1,:,:] + bwd_flow[:,1,:,:]).unsqueeze(1)], 1)
        bwd_match = torch.cat([grid, bwd_corres], 1) # [b,4,h,w]

        # Use fwd-bwd consistency map for filter
        img1_score_mask = img1_valid_mask * 1.0 / (0.1+img1_flow_diff_mask.mean(1).unsqueeze(1))
        img2_score_mask = img2_valid_mask * 1.0 / (0.1+img2_flow_diff_mask.mean(1).unsqueeze(1))
        # img1_score_mask = img1_valid_mask
        
        F_final_1 = self.filter(fwd_match, img1_score_mask, visualizer=visualizer)
        _, dist_map_1 = self.compute_epipolar_loss(F_final_1, fwd_match.view([batch_size,4,-1]), img1_valid_mask.view([batch_size,1,-1]))
        dist_map_1 = dist_map_1.view([batch_size, img_h, img_w, 1])

        # Compute geo loss for regularize correspondence.
        rigid_mask_1, inlier_mask_1, rigid_score_1 = self.get_rigid_mask(dist_map_1)
        
        # We only use rigid mask to filter out the moving objects for computing geo loss.
        geo_loss = (dist_map_1 * (rigid_mask_1 - inlier_mask_1)).mean((1,2,3)) / \
             (rigid_mask_1 - inlier_mask_1).mean((1,2,3))

        loss_pack['geo_loss'] = geo_loss
        
        if output_F:
            return loss_pack, F_final_1, img1_score_mask, rigid_score_1.permute(0,3,1,2), fwd_flow, fwd_match
        else:
            return loss_pack




