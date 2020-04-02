import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from structures import *
from model_triangulate_pose import Model_triangulate_pose
from pytorch_ssim import SSIM
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'visualize'))
from visualizer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Model_depth_pose(nn.Module):
    def __init__(self, cfg):
        super(Model_depth_pose, self).__init__()
        self.depth_match_num = cfg.depth_match_num
        self.depth_sample_ratio = cfg.depth_sample_ratio
        self.depth_scale = cfg.depth_scale
        self.w_flow_error = cfg.w_flow_error
        self.dataset = cfg.dataset

        self.depth_net = Depth_Model(cfg.depth_scale)
        self.model_pose = Model_triangulate_pose(cfg)

    def meshgrid(self, h, w):
        xx, yy = np.meshgrid(np.arange(0,w), np.arange(0,h))
        meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
        meshgrid = torch.from_numpy(meshgrid)
        return meshgrid
    
    def robust_rand_sample(self, match, mask, num):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, n = match.shape[0], match.shape[2]
        nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1)) # []
        if nonzeros_num.detach().cpu().numpy() == n:
            rand_int = torch.randint(0, n, [num])
            select_match = match[:,:,rand_int]
        else:
            # If there is zero score in match, sample the non-zero matches.
            num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
            select_idxs = []
            for i in range(b):
                nonzero_idx = torch.nonzero(mask[i,0,:]) # [nonzero_num,1]
                rand_int = torch.randint(0, nonzero_idx.shape[0], [int(num)])
                select_idx = nonzero_idx[rand_int, :] # [num, 1]
                select_idxs.append(select_idx)
            select_idxs = torch.stack(select_idxs, 0) # [b,num,1]
            select_match = torch.gather(match.transpose(1,2), index=select_idxs.repeat(1,1,4), dim=1).transpose(1,2) # [b, 4, num]
        return select_match, num

    def top_ratio_sample(self, match, mask, ratio):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, total_num = match.shape[0], match.shape[-1]
        scores, indices = torch.topk(mask, int(ratio*total_num), dim=-1) # [B, 1, ratio*tnum]
        select_match = torch.gather(match.transpose(1,2), index=indices.squeeze(1).unsqueeze(-1).repeat(1,1,4), dim=1).transpose(1,2) # [b, 4, ratio*tnum]
        return select_match, scores
    
    def rand_sample(self, match, num):
        b, c, n = match.shape[0], match.shape[1], match.shape[2]
        rand_int = torch.randint(0, match.shape[-1], size=[num])
        select_pts = match[:,:,rand_int]
        return select_pts
    
    def filt_negative_depth(self, point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord):
        # Filter out the negative projection depth.
        # point2d_1_depth: [b, n, 1]
        b, n = point2d_1_depth.shape[0], point2d_1_depth.shape[1]
        mask = (point2d_1_depth > 0.01).float() * (point2d_2_depth > 0.01).float()
        select_idxs = []
        flag = 0
        for i in range(b):
            if torch.sum(mask[i,:,0]) == n:
                idx = torch.arange(n).to(mask.get_device())
            else:
                nonzero_idx = torch.nonzero(mask[i,:,0]).squeeze(1) # [k]
                if nonzero_idx.shape[0] < 0.1*n:
                    idx = torch.arange(n).to(mask.get_device())
                    flag = 1
                else:
                    res = torch.randint(0, nonzero_idx.shape[0], size=[n-nonzero_idx.shape[0]]).to(mask.get_device()) # [n-nz]
                    idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
            select_idxs.append(idx)
        select_idxs = torch.stack(select_idxs, dim=0) # [b,n]
        point2d_1_depth = torch.gather(point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1) # [b,n,1]
        point2d_2_depth = torch.gather(point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1) # [b,n,1]
        point2d_1_coord = torch.gather(point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1,1,2), dim=1) # [b,n,2]
        point2d_2_coord = torch.gather(point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1,1,2), dim=1) # [b,n,2]
        return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag
    
    def filt_invalid_coord(self, point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, max_h, max_w):
        # Filter out the negative projection depth.
        # point2d_1_depth: [b, n, 1]
        b, n = point2d_1_coord.shape[0], point2d_1_coord.shape[1]
        max_coord = torch.Tensor([max_w, max_h]).to(point2d_1_coord.get_device())
        mask = (point2d_1_coord > 0).all(dim=-1, keepdim=True).float() * (point2d_2_coord > 0).all(dim=-1, keepdim=True).float() * \
            (point2d_1_coord < max_coord).all(dim=-1, keepdim=True).float() * (point2d_2_coord < max_coord).all(dim=-1, keepdim=True).float()
        
        flag = 0
        if torch.sum(1.0-mask) == 0:
            return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

        select_idxs = []
        for i in range(b):
            if torch.sum(mask[i,:,0]) == n:
                idx = torch.arange(n).to(mask.get_device())
            else:
                nonzero_idx = torch.nonzero(mask[i,:,0]).squeeze(1) # [k]
                if nonzero_idx.shape[0] < 0.1*n:
                    idx = torch.arange(n).to(mask.get_device())
                    flag = 1
                else:
                    res = torch.randint(0, nonzero_idx.shape[0], size=[n-nonzero_idx.shape[0]]).to(mask.get_device())
                    idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
            select_idxs.append(idx)
        select_idxs = torch.stack(select_idxs, dim=0) # [b,n]
        point2d_1_depth = torch.gather(point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1) # [b,n,1]
        point2d_2_depth = torch.gather(point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1) # [b,n,1]
        point2d_1_coord = torch.gather(point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1,1,2), dim=1) # [b,n,2]
        point2d_2_coord = torch.gather(point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1,1,2), dim=1) # [b,n,2]
        return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag
    
    def ray_angle_filter(self, match, P1, P2, return_angle=False):
        # match: [b, 4, n] P: [B, 3, 4]
        b, n = match.shape[0], match.shape[2]
        K = P1[:,:,:3] # P1 with identity rotation and zero translation
        K_inv = torch.inverse(K)
        RT1 = K_inv.bmm(P1) # [b, 3, 4]
        RT2 = K_inv.bmm(P2)
        ones = torch.ones([b,1,n]).to(match.get_device())
        pts1 = torch.cat([match[:,:2,:], ones], 1)
        pts2 = torch.cat([match[:,2:,:], ones], 1)
        
        ray1_dir = (RT1[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts1)# [b,3,n]
        ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray1_origin = (-1) * RT1[:,:,:3].transpose(1,2).bmm(RT1[:,:,3].unsqueeze(-1)) # [b, 3, 1]
        ray2_dir = (RT2[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts2) # [b,3,n]
        ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray2_origin = (-1) * RT2[:,:,:3].transpose(1,2).bmm(RT2[:,:,3].unsqueeze(-1)) # [b, 3, 1]

        # We compute the angle betwwen vertical line from ray1 origin to ray2 and ray1.
        p1p2 = (ray1_origin - ray2_origin).repeat(1,1,n)
        verline = ray2_origin.repeat(1,1,n) + torch.sum(p1p2 * ray2_dir, dim=1, keepdim=True) * ray2_dir - ray1_origin.repeat(1,1,n) # [b,3,n]
        cosvalue = torch.sum(ray1_dir * verline, dim=1, keepdim=True)  / \
            ((torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12) * (torch.norm(verline, dim=1, keepdim=True, p=2) + 1e-12))# [b,1,n]

        mask = (cosvalue > 0.001).float() # we drop out angles less than 1' [b,1,n]
        flag = 0
        num = torch.min(torch.sum(mask, -1)).int()
        if num.cpu().detach().numpy() == 0:
            flag = 1
            filt_match = match[:,:,:100]
            if return_angle:
                return filt_match, flag, torch.zeros_like(mask).to(filt_match.get_device())
            else:
                return filt_match, flag
        nonzero_idx = []
        for i in range(b):
            idx = torch.nonzero(mask[i,0,:])[:num] # [num,1]
            nonzero_idx.append(idx)
        nonzero_idx = torch.stack(nonzero_idx, 0) # [b,num,1]
        filt_match = torch.gather(match.transpose(1,2), index=nonzero_idx.repeat(1,1,4), dim=1).transpose(1,2) # [b,4,num]
        if return_angle:
            return filt_match, flag, mask
        else:
            return filt_match, flag
    
    def midpoint_triangulate(self, match, K_inv, P1, P2):
        # match: [b, 4, num] P1: [b, 3, 4]
        # Match is in the image coordinates. P1, P2 is camera parameters. [B, 3, 4] match: [B, M, 4]
        b, n = match.shape[0], match.shape[2]
        RT1 = K_inv.bmm(P1) # [b, 3, 4]
        RT2 = K_inv.bmm(P2)
        ones = torch.ones([b,1,n]).to(match.get_device())
        pts1 = torch.cat([match[:,:2,:], ones], 1)
        pts2 = torch.cat([match[:,2:,:], ones], 1)
        
        ray1_dir = (RT1[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts1)# [b,3,n]
        ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray1_origin = (-1) * RT1[:,:,:3].transpose(1,2).bmm(RT1[:,:,3].unsqueeze(-1)) # [b, 3, 1]
        ray2_dir = (RT2[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts2) # [b,3,n]
        ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray2_origin = (-1) * RT2[:,:,:3].transpose(1,2).bmm(RT2[:,:,3].unsqueeze(-1)) # [b, 3, 1]
    
        dir_cross = torch.cross(ray1_dir, ray2_dir, dim=1) # [b,3,n]
        denom = 1.0 / (torch.sum(dir_cross * dir_cross, dim=1, keepdim=True)+1e-12) # [b,1,n]
        origin_vec = (ray2_origin - ray1_origin).repeat(1,1,n) # [b,3,n]
        a1 = origin_vec.cross(ray2_dir, dim=1) # [b,3,n]
        a1 = torch.sum(a1 * dir_cross, dim=1, keepdim=True) * denom # [b,1,n]
        a2 = origin_vec.cross(ray1_dir, dim=1) # [b,3,n]
        a2 = torch.sum(a2 * dir_cross, dim=1, keepdim=True) * denom # [b,1,n]
        p1 = ray1_origin + a1 * ray1_dir
        p2 = ray2_origin + a2 * ray2_dir
        point = (p1 + p2) / 2.0 # [b,3,n]
        # Convert to homo coord to get consistent with other functions.
        point_homo = torch.cat([point, ones], dim=1).transpose(1,2) # [b,n,4]
        return point_homo
    
    def rt_from_fundamental_mat_nyu(self, fmat, K, depth_match):
        # F: [b, 3, 3] K: [b, 3, 3] depth_match: [b ,4, n]
        #verify_match = self.rand_sample(depth_match, 5000) # [b,4,100]
        verify_match = depth_match.transpose(1,2).cpu().detach().numpy()
        K_inv = torch.inverse(K)
        b = fmat.shape[0]
        fmat_ = K.transpose(1,2).bmm(fmat)
        essential_mat = fmat_.bmm(K)
        iden = torch.cat([torch.eye(3), torch.zeros([3,1])], -1).unsqueeze(0).repeat(b,1,1).to(K.get_device()) # [b,3,4]
        P1 = K.bmm(iden)
        flags = []
        number_inliers = []
        P2 = []
        for i in range(b):
            cnum, R, t, _ = cv2.recoverPose(essential_mat[i].cpu().detach().numpy().astype('float64'), verify_match[i,:,:2].astype('float64'), \
                verify_match[i,:,2:].astype('float64'), cameraMatrix=K[i,:,:].cpu().detach().numpy().astype('float64'))
            p2 = torch.from_numpy(np.concatenate([R, t], axis=-1)).float().to(K.get_device())
            P2.append(p2)
            if cnum > depth_match.shape[-1] / 7.0:
                flags.append(1)
            else:
                flags.append(0)
            number_inliers.append(cnum)
            
        P2 = K.bmm(torch.stack(P2, axis=0))
        #pdb.set_trace()
        
        return P1, P2, flags
    
    def verifyRT(self, match, K_inv, P1, P2):
        # match: [b, 4, n] P1: [b,3,4] P2: [b,3,4]
        b, n = match.shape[0], match.shape[2]
        point3d = self.midpoint_triangulate(match, K_inv, P1, P2).reshape([-1,4]).unsqueeze(-1) # [b*n, 4, 1]
        P1_ = P1.repeat(n,1,1)
        P2_ = P2.repeat(n,1,1)
        depth1 = P1_.bmm(point3d)[:,-1,:] / point3d[:,-1,:] # [b*n, 1]
        depth2 = P2_.bmm(point3d)[:,-1,:] / point3d[:,-1,:]
        inlier_num = torch.sum((depth1.view([b,n]) > 0).float() * (depth2.view([b,n]) > 0).float(), 1) # [b]
        return inlier_num
    
    def rt_from_fundamental_mat(self, fmat, K, depth_match):
        # F: [b, 3, 3] K: [b, 3, 3] depth_match: [b ,4, n]
        verify_match = self.rand_sample(depth_match, 200) # [b,4,100]
        K_inv = torch.inverse(K)
        b = fmat.shape[0]
        fmat_ = K.transpose(1,2).bmm(fmat)
        essential_mat = fmat_.bmm(K)
        essential_mat_cpu = essential_mat.cpu()
        U, S, V = torch.svd(essential_mat_cpu)
        U, S, V = U.to(K.get_device()), S.to(K.get_device()), V.to(K.get_device())
        W = torch.from_numpy(np.array([[[0., -1., 0.],[1., 0., 0.],[0., 0., 1.]]])).float().repeat(b,1,1).to(K.get_device())
        # R = UWV^t or UW^tV^t t = U[:,2] the third column of U
        R1 = U.bmm(W).bmm(V.transpose(1,2)) # Do we need matrix determinant sign?
        R1 = torch.sign(torch.det(R1)).unsqueeze(-1).unsqueeze(-1) * R1
        R2 = U.bmm(W.transpose(1,2)).bmm(V.transpose(1,2))
        R2 = torch.sign(torch.det(R2)).unsqueeze(-1).unsqueeze(-1) * R2
        t1 = U[:,:,2].unsqueeze(-1) # The third column
        t2 = -U[:,:,2].unsqueeze(-1) # Inverse direction
        
        iden = torch.cat([torch.eye(3), torch.zeros([3,1])], -1).unsqueeze(0).repeat(b,1,1).to(K.get_device()) # [b,3,4]
        P1 = K.bmm(iden)
        P2_1 = K.bmm(torch.cat([R1, t1], -1))
        P2_2 = K.bmm(torch.cat([R2, t1], -1))
        P2_3 = K.bmm(torch.cat([R1, t2], -1))
        P2_4 = K.bmm(torch.cat([R2, t2], -1))
        P2_c = [P2_1, P2_2, P2_3, P2_4]
        flags = []
        for i in range(4):
            with torch.no_grad():
                inlier_num = self.verifyRT(verify_match, K_inv, P1, P2_c[i])
                flags.append(inlier_num)
        P2_c = torch.stack(P2_c, dim=1) # [B, 4, 3, 4]
        flags = torch.stack(flags, dim=1) # [B, 4]
        idx = torch.argmax(flags, dim=-1, keepdim=True) # [b,1]
        P2 = torch.gather(P2_c, index=idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3,4), dim=1).squeeze(1) # [b,3,4]
        #pdb.set_trace()
        return P1, P2
    
    def reproject(self, P, point3d):
        # P: [b,3,4] point3d: [b,n,4]
        point2d = P.bmm(point3d.transpose(1,2)) # [b,4,n]
        point2d_coord = (point2d[:,:2,:] / (point2d[:,2,:].unsqueeze(1) + 1e-12)).transpose(1,2) # [b,n,2]
        point2d_depth = point2d[:,2,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        return point2d_coord, point2d_depth

    def scale_adapt(self, depth1, depth2, eps=1e-12):
        with torch.no_grad():
            A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1) # [b,1]
            C = torch.sum(depth1 / (depth2 + eps), dim=1) # [b,1]
            a = C / (A + eps)
        return a

    def affine_adapt(self, depth1, depth2, use_translation=True, eps=1e-12):
        a_scale = self.scale_adapt(depth1, depth2, eps=eps)
        if not use_translation: # only fit the scale parameter
            return a_scale, torch.zeros_like(a_scale)
        else:
            with torch.no_grad():
                A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1) # [b,1]
                B = torch.sum(depth1 / (depth2 ** 2 + eps), dim=1) # [b,1]
                C = torch.sum(depth1 / (depth2 + eps), dim=1) # [b,1]
                D = torch.sum(1.0 / (depth2 ** 2 + eps), dim=1) # [b,1]
                E = torch.sum(1.0 / (depth2 + eps), dim=1) # [b,1]
                a = (B*E - D*C) / (B*B - A*D + 1e-12)
                b = (B*C - A*E) / (B*B - A*D + 1e-12)

                # check ill condition
                cond = (B*B - A*D)
                valid = (torch.abs(cond) > 1e-4).float()
                a = a * valid + a_scale * (1 - valid)
                b = b * valid
            return a, b

    def register_depth(self, depth_pred, coord_tri, depth_tri):
        # depth_pred: [b, 1, h, w] coord_tri: [b,n,2] depth_tri: [b,n,1]
        batch, _, h, w = depth_pred.shape[0], depth_pred.shape[1], depth_pred.shape[2], depth_pred.shape[3]
        n = depth_tri.shape[1]
        coord_tri_nor = torch.stack([2.0*coord_tri[:,:,0] / (w-1.0) - 1.0, 2.0*coord_tri[:,:,1] / (h-1.0) - 1.0], -1)
        depth_inter = F.grid_sample(depth_pred, coord_tri_nor.view([batch,n,1,2]), padding_mode='reflection').squeeze(-1).transpose(1,2) # [b,n,1]

        # Normalize
        scale = torch.median(depth_inter, 1)[0] / (torch.median(depth_tri, 1)[0] + 1e-12)
        scale = scale.detach() # [b,1]
        scale_depth_inter = depth_inter / (scale.unsqueeze(-1) + 1e-12)
        scale_depth_pred = depth_pred / (scale.unsqueeze(-1).unsqueeze(-1) + 1e-12)
        
        # affine adapt
        a, b = self.affine_adapt(scale_depth_inter, depth_tri, use_translation=False)
        affine_depth_inter = a.unsqueeze(1) * scale_depth_inter + b.unsqueeze(1) # [b,n,1]
        affine_depth_pred = a.unsqueeze(-1).unsqueeze(-1) * scale_depth_pred + b.unsqueeze(-1).unsqueeze(-1) # [b,1,h,w]
        return affine_depth_pred, affine_depth_inter
    
    def get_trian_loss(self, tri_depth, pred_tri_depth):
        # depth: [b,n,1]
        loss = torch.pow(1.0 - pred_tri_depth / (tri_depth + 1e-12), 2).mean((1,2))
        return loss
    
    def get_reproj_fdp_loss(self, pred1, pred2, P2, K, K_inv, valid_mask, rigid_mask, flow, visualizer=None):
        # pred: [b,1,h,w] Rt: [b,3,4] K: [b,3,3] mask: [b,1,h,w] flow: [b,2,h,w]
        b, h, w = pred1.shape[0], pred1.shape[2], pred1.shape[3]
        xy = self.meshgrid(h,w).unsqueeze(0).repeat(b,1,1,1).float().to(flow.get_device()) # [b,2,h,w]
        ones = torch.ones([b,1,h,w]).float().to(flow.get_device())
        pts1_3d = K_inv.bmm(torch.cat([xy, ones], 1).view([b,3,-1])) * pred1.view([b,1,-1]) # [b,3,h*w]
        pts2_coord, pts2_depth = self.reproject(P2, torch.cat([pts1_3d, ones.view([b,1,-1])], 1).transpose(1,2)) # [b,h*w, 2]
        # TODO Here some of the reprojection coordinates are invalid. (<0 or >max)
        reproj_valid_mask = (pts2_coord > torch.Tensor([0,0]).to(pred1.get_device())).all(-1, True).float() * \
            (pts2_coord < torch.Tensor([w-1,h-1]).to(pred1.get_device())).all(-1, True).float() # [b,h*w, 1]
        reproj_valid_mask = (valid_mask * reproj_valid_mask.view([b,h,w,1]).permute([0,3,1,2])).detach()
        rigid_mask = rigid_mask.detach()
        pts2_depth = pts2_depth.transpose(1,2).view([b,1,h,w])

        # Get the interpolated depth prediction2
        pts2_coord_nor = torch.cat([2.0 * pts2_coord[:,:,0].unsqueeze(-1) / (w - 1.0) - 1.0, 2.0 * pts2_coord[:,:,1].unsqueeze(-1) / (h - 1.0) - 1.0], -1)
        inter_depth2 = F.grid_sample(pred2, pts2_coord_nor.view([b, h, w, 2]), padding_mode='reflection') # [b,1,h,w]
        pj_loss_map = (torch.abs(1.0 - pts2_depth / (inter_depth2 + 1e-12)) * rigid_mask * reproj_valid_mask)
        pj_loss = pj_loss_map.mean((1,2,3)) / ((reproj_valid_mask * rigid_mask).mean((1,2,3))+1e-12)
        #pj_loss = (valid_mask * mask * torch.abs(pts2_depth - inter_depth2) / (torch.abs(pts2_depth + inter_depth2)+1e-12)).mean((1,2,3)) / ((valid_mask * mask).mean((1,2,3))+1e-12) # [b]
        flow_loss = (rigid_mask * torch.abs(flow + xy - pts2_coord.detach().permute(0,2,1).view([b,2,h,w]))).mean((1,2,3)) / (rigid_mask.mean((1,2,3)) + 1e-12)
        return pj_loss, flow_loss
    
    def disp2depth(self, disp, min_depth=0.1, max_depth=100.0):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth
    
    def get_smooth_loss(self, img, disp):
        # img: [b,3,h,w] depth: [b,1,h,w]
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean((1,2,3)) + grad_disp_y.mean((1,2,3))

    def infer_depth(self, img):
        disp_list = self.depth_net(img)
        disp, depth = self.disp2depth(disp_list[0])
        return disp_list[0]
    
    def infer_vo(self, img1, img2, K, K_inv, match_num=6000):
        b, img_h, img_w = img1.shape[0], img1.shape[2], img1.shape[3]
        F_final, img1_valid_mask, img1_rigid_mask, fwd_flow, fwd_match = self.model_pose.inference(img1, img2, K, K_inv)
        # infer depth
        disp1_list = self.depth_net(img1) # Nscales * [B, 1, H, W]
        disp2_list = self.depth_net(img2)
        disp1, depth1 = self.disp2depth(disp1_list[0])
        disp2, depth2 = self.disp2depth(disp2_list[0])

        img1_depth_mask = img1_rigid_mask * img1_valid_mask
        # [b, 4, match_num]
        top_ratio_match, top_ratio_mask = self.top_ratio_sample(fwd_match.view([b,4,-1]), img1_depth_mask.view([b,1,-1]), ratio=0.30) # [b, 4, ratio*h*w]
        depth_match, depth_match_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask, num=match_num)
        return depth_match, depth1, depth2
    
    def check_rt(self, img1, img2, K, K_inv):
        # initialization
        b = img1.shape[0]
        flag1, flag2, flag3 = 0, 0, 0
        images = torch.cat([img1, img2], dim=2)
        inputs = [images, K.unsqueeze(1), K_inv.unsqueeze(1)]

        # Pose Network
        #self.profiler.reset()
        loss_pack, F_final, img1_valid_mask, img1_rigid_score, img1_inlier_mask, fwd_flow, fwd_match = self.model_pose(inputs, output_F=True)
        # Get masks
        img1_depth_mask = img1_rigid_score * img1_valid_mask

        # Select top score matches to triangulate depth.
        top_ratio_match, top_ratio_mask = self.top_ratio_sample(fwd_match.view([b,4,-1]), img1_depth_mask.view([b,1,-1]), ratio=0.20) # [b, 4, ratio*h*w]
        depth_match, depth_match_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask, num=self.depth_match_num)

        P1, P2, flags = self.rt_from_fundamental_mat_nyu(F_final.detach(), K, depth_match)
        P1 = P1.detach()
        P2 = P2.detach()
        flags = torch.from_numpy(np.stack(flags, axis=0)).float().to(K.get_device())

        return flags

    def inference(self, img1, img2, K, K_inv):
        b, img_h, img_w = img1.shape[0], img1.shape[2], img1.shape[3]
        visualizer = Visualizer_debug('./vis/', np.transpose(255*img1.detach().cpu().numpy(), [0,2,3,1]), \
            np.transpose(255*img2.detach().cpu().numpy(), [0,2,3,1]))

        F_final, img1_valid_mask, img1_rigid_mask, fwd_flow, fwd_match = self.model_pose.inference(img1, img2, K, K_inv)
        # infer depth
        disp1_list = self.depth_net(img1) # Nscales * [B, 1, H, W]
        disp2_list = self.depth_net(img2)
        disp1, _ = self.disp2depth(disp1_list[0])
        disp2, _ = self.disp2depth(disp2_list[0])

        # Get Camera Matrix
        img1_depth_mask = img1_rigid_mask * img1_valid_mask
        # [b, 4, match_num]
        top_ratio_match, top_ratio_mask = self.top_ratio_sample(fwd_match.view([b,4,-1]), img1_depth_mask.view([b,1,-1]), ratio=0.20) # [b, 4, ratio*h*w]
        depth_match, depth_match_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask, num=self.depth_match_num)
        if self.dataset == 'nyuv2':
            P1, P2, _ = self.rt_from_fundamental_mat_nyu(F_final, K, depth_match)
        else:
            P1, P2 = self.rt_from_fundamental_mat(F_final, K, depth_match)
        Rt = K_inv.bmm(P2)
        
        filt_depth_match, flag1 = self.ray_angle_filter(depth_match, P1, P2) # [b, 4, filt_num]
        point3d_1 = self.midpoint_triangulate(filt_depth_match, K_inv, P1, P2)
        point2d_1_coord, point2d_1_depth = self.reproject(P1, point3d_1) # [b,n,2], [b,n,1]
        point2d_2_coord, point2d_2_depth = self.reproject(P2, point3d_1)
        
        # Filter out some invalid triangulation results to stablize training.
        point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag2 = self.filt_negative_depth(point2d_1_depth, \
                point2d_2_depth, point2d_1_coord, point2d_2_coord)
        point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag3 = self.filt_invalid_coord(point2d_1_depth, \
                point2d_2_depth, point2d_1_coord, point2d_2_coord, max_h=img_h, max_w=img_w)
        
        return fwd_flow, disp1, disp2, Rt, point2d_1_coord, point2d_1_depth
    

    def forward(self, inputs):
        # initialization
        images, K_ms, K_inv_ms = inputs
        K, K_inv = K_ms[:,0,:,:], K_inv_ms[:,0,:,:]
        assert (images.shape[1] == 3)
        img_h, img_w = int(images.shape[2] / 2), images.shape[3] 
        img1, img2 = images[:,:,:img_h,:], images[:,:,img_h:,:]
        b = img1.shape[0]
        flag1, flag2, flag3 = 0, 0, 0
        visualizer = Visualizer_debug('./vis/', img1=255*img1.permute([0,2,3,1]).detach().cpu().numpy(), \
            img2=255*img2.permute([0,2,3,1]).detach().cpu().numpy())
        
        # Pose Network
        loss_pack, F_final, img1_valid_mask, img1_rigid_mask, fwd_flow, fwd_match = self.model_pose(inputs, output_F=True, visualizer=visualizer)
        # infer depth
        disp1_list = self.depth_net(img1) # Nscales * [B, 1, H, W]
        disp2_list = self.depth_net(img2)

        # Get masks
        img1_depth_mask = img1_rigid_mask * img1_valid_mask
        
        # Select top score matches to triangulate depth.
        top_ratio_match, top_ratio_mask = self.top_ratio_sample(fwd_match.view([b,4,-1]), img1_depth_mask.view([b,1,-1]), ratio=self.depth_sample_ratio) # [b, 4, ratio*h*w]
        depth_match, depth_match_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask, num=self.depth_match_num)
        
        if self.dataset == 'nyuv2':
            P1, P2, flags = self.rt_from_fundamental_mat_nyu(F_final.detach(), K, depth_match)
            flags = torch.from_numpy(np.stack(flags, axis=0)).float().to(K.get_device())
        else:
            P1, P2 = self.rt_from_fundamental_mat(F_final.detach(), K, depth_match)
        P1 = P1.detach()
        P2 = P2.detach()

        # Get triangulated points
        filt_depth_match, flag1 = self.ray_angle_filter(depth_match, P1, P2, return_angle=False) # [b, 4, filt_num]
        
        point3d_1 = self.midpoint_triangulate(filt_depth_match, K_inv, P1, P2)
        point2d_1_coord, point2d_1_depth = self.reproject(P1, point3d_1) # [b,n,2], [b,n,1]
        point2d_2_coord, point2d_2_depth = self.reproject(P2, point3d_1)
        
        # Filter out some invalid triangulation results to stablize training.
        point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag2 = self.filt_negative_depth(point2d_1_depth, \
                point2d_2_depth, point2d_1_coord, point2d_2_coord)
        point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag3 = self.filt_invalid_coord(point2d_1_depth, \
                point2d_2_depth, point2d_1_coord, point2d_2_coord, max_h=img_h, max_w=img_w)

        
        if flag1 + flag2 + flag3 > 0:
            loss_pack['pt_depth_loss'] = torch.zeros([2]).to(point3d_1.get_device()).requires_grad_()
            loss_pack['pj_depth_loss'] = torch.zeros([2]).to(point3d_1.get_device()).requires_grad_()
            loss_pack['flow_error'] = torch.zeros([2]).to(point3d_1.get_device()).requires_grad_()
            loss_pack['depth_smooth_loss'] = torch.zeros([2]).to(point3d_1.get_device()).requires_grad_()
            return loss_pack

        pt_depth_loss = 0
        pj_depth_loss = 0
        flow_error = 0
        depth_smooth_loss = 0
        for s in range(self.depth_scale):
            disp_pred1 = F.interpolate(disp1_list[s], size=(img_h, img_w), mode='bilinear') # [b,1,h,w]
            disp_pred2 = F.interpolate(disp2_list[s], size=(img_h, img_w), mode='bilinear')
            scaled_disp1, depth_pred1 = self.disp2depth(disp_pred1)
            scaled_disp2, depth_pred2 = self.disp2depth(disp_pred2)
            # Rescale predicted depth according to triangulated depth
            # [b,1,h,w], [b,n,1]
            rescaled_pred1, inter_pred1 = self.register_depth(depth_pred1, point2d_1_coord, point2d_1_depth)
            rescaled_pred2, inter_pred2 = self.register_depth(depth_pred2, point2d_2_coord, point2d_2_depth)
            # Get Losses
            
            pt_depth_loss += self.get_trian_loss(point2d_1_depth, inter_pred1) + self.get_trian_loss(point2d_2_depth, inter_pred2)
            pj_depth, flow_loss = self.get_reproj_fdp_loss(rescaled_pred1, rescaled_pred2, P2, K, K_inv, img1_valid_mask, img1_rigid_mask, fwd_flow, visualizer=visualizer)
            depth_smooth_loss += self.get_smooth_loss(img1, disp_pred1 / (disp_pred1.mean((2,3), True) + 1e-12)) + \
                self.get_smooth_loss(img2, disp_pred2 / (disp_pred2.mean((2,3), True) + 1e-12))
            pj_depth_loss += pj_depth
            flow_error += flow_loss

        if self.dataset == 'nyuv2':
            loss_pack['pt_depth_loss'] = pt_depth_loss * flags
            loss_pack['pj_depth_loss'], loss_pack['flow_error'] = pj_depth_loss * flags, flow_error * flags
            loss_pack['depth_smooth_loss'] = depth_smooth_loss * flags
        else:
            loss_pack['pt_depth_loss'] = pt_depth_loss
            loss_pack['pj_depth_loss'], loss_pack['flow_error'] = pj_depth_loss, flow_error
            loss_pack['depth_smooth_loss'] = depth_smooth_loss
        return loss_pack


