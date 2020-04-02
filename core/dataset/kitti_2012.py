import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kitti_prepared import KITTI_Prepared
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation'))
from evaluate_flow import get_scaled_intrinsic_matrix, eval_flow_avg
import numpy as np
import cv2
import copy

import torch
import pdb

class KITTI_2012(KITTI_Prepared):
    def __init__(self, data_dir, img_hw=(256, 832), init=True):
        self.data_dir = data_dir
        self.img_hw = img_hw
        self.num_total = 194
        if init:
            self.data_list = self.get_data_list()

    def get_data_list(self):
        data_list = []
        for i in range(self.num_total):
            data = {}
            data['img1_dir'] = os.path.join(self.data_dir, 'image_2', str(i).zfill(6) + '_10.png')
            data['img2_dir'] = os.path.join(self.data_dir, 'image_2', str(i).zfill(6) + '_11.png')
            data['calib_file_dir'] = os.path.join(self.data_dir, 'calib_cam_to_cam', str(i).zfill(6) + '.txt')
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def read_cam_intrinsic(self, calib_file):
        input_intrinsic = get_scaled_intrinsic_matrix(calib_file, zoom_x=1.0, zoom_y=1.0)
        return input_intrinsic

    def __getitem__(self, idx):
        '''
        Returns:
        - img		torch.Tensor (N * H, W, 3)
        - K	torch.Tensor (num_scales, 3, 3)
        - K_inv	torch.Tensor (num_scales, 3, 3)
        '''
        data = self.data_list[idx]
        # load img
        img1 = cv2.imread(data['img1_dir'])
        img2 = cv2.imread(data['img2_dir'])
        img_hw_orig = (img1.shape[0], img1.shape[1])
        img = np.concatenate([img1, img2], 0)
        img = self.preprocess_img(img, self.img_hw, is_test=True)
        img  = img.transpose(2,0,1)

        # load intrinsic
        cam_intrinsic = self.read_cam_intrinsic(data['calib_file_dir'])
        cam_intrinsic = self.rescale_intrinsics(cam_intrinsic, img_hw_orig, self.img_hw)
        K, K_inv = self.get_intrinsics_per_scale(cam_intrinsic, scale=0) # (3, 3), (3, 3)
        return torch.from_numpy(img).float(), torch.from_numpy(K).float(), torch.from_numpy(K_inv).float()

if __name__ == '__main__':
    pass

