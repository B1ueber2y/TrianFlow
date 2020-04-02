import torch
import numpy as np
import os, sys
import torch.nn as nn
import pdb
import cv2

class reduced_ransac(nn.Module):
    def __init__(self, check_num, thres, dataset):
        super(reduced_ransac, self).__init__()
        self.check_num = check_num
        self.thres = thres
        self.dataset = dataset

    def robust_rand_sample(self, match, mask, num, robust=True):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, n = match.shape[0], match.shape[2]
        nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1)) # []
        if nonzeros_num.detach().cpu().numpy() == n:
            rand_int = torch.randint(0, n, [num])
            select_match = match[:,:,rand_int]
        else:
            # If there is zero score in match, sample the non-zero matches.
            select_idxs = []
            if robust:
                num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
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


    def forward(self, match, mask, visualizer=None):
        # match: [B, 4, H, W] mask: [B, 1, H, W]
        b, h, w = match.shape[0], match.shape[2], match.shape[3]
        match = match.view([b, 4, -1]).contiguous()
        mask = mask.view([b, 1, -1]).contiguous()
        
        # Sample matches for RANSAC 8-point and best F selection
        top_ratio_match, top_ratio_mask = self.top_ratio_sample(match, mask, ratio=0.20) # [b, 4, ratio*H*W] 
        check_match, check_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask, num=self.check_num) # [b, 4, check_num]
        check_match = check_match.contiguous()

        cv_f = []
        for i in range(b):
            if self.dataset == 'nyuv2':
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_LMEDS, 0.99)
            else:
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_RANSAC, 0.1, 0.99)
            cv_f.append(f)
        cv_f = np.stack(cv_f, axis=0)
        cv_f = torch.from_numpy(cv_f).float().to(match.get_device())
        
        return cv_f

