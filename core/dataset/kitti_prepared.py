import os, sys
import numpy as np
import cv2
import copy

import torch
import torch.utils.data
import pdb

class KITTI_Prepared(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_scales=3, img_hw=(256, 832), num_iterations=None):
        super(KITTI_Prepared, self).__init__()
        self.data_dir = data_dir
        self.num_scales = num_scales
        self.img_hw = img_hw
        self.num_iterations = num_iterations

        info_file = os.path.join(self.data_dir, 'train.txt')
        #info_file = os.path.join(self.data_dir, 'train_flow.txt')
        self.data_list = self.get_data_list(info_file)

    def get_data_list(self, info_file):
        with open(info_file, 'r') as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            k = line.strip('\n').split()
            data = {}
            data['image_file'] = os.path.join(self.data_dir, k[0])
            data['cam_intrinsic_file'] = os.path.join(self.data_dir, k[1])
            data_list.append(data)
        print('A total of {} image pairs found'.format(len(data_list)))
        return data_list

    def count(self):
        return len(self.data_list)

    def rand_num(self, idx):
        num_total = self.count()
        np.random.seed(idx)
        num = np.random.randint(num_total)
        return num

    def __len__(self):
        if self.num_iterations is None:
            return self.count()
        else:
            return self.num_iterations

    def resize_img(self, img, img_hw):
        '''
        Input size (N*H, W, 3)
        Output size (N*H', W', 3), where (H', W') == self.img_hw
        '''
        img_h, img_w = img.shape[0], img.shape[1]
        img_hw_orig = (int(img_h / 2), img_w) 
        img1, img2 = img[:img_hw_orig[0], :, :], img[img_hw_orig[0]:, :, :]
        img1_new = cv2.resize(img1, (img_hw[1], img_hw[0]))
        img2_new = cv2.resize(img2, (img_hw[1], img_hw[0]))
        img_new = np.concatenate([img1_new, img2_new], 0)
        return img_new

    def random_flip_img(self, img):
        is_flip = (np.random.rand() > 0.5)
        if is_flip:
            img = cv2.flip(img, 1)
        return img

    def preprocess_img(self, img, img_hw=None, is_test=False):
        if img_hw is None:
            img_hw = self.img_hw
        img = self.resize_img(img, img_hw)
        if not is_test:
            img = self.random_flip_img(img)
        img = img / 255.0
        return img

    def read_cam_intrinsic(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        data = lines[-1].strip('\n').split(' ')[1:]
        data = [float(k) for k in data]
        data = np.array(data).reshape(3,4)
        cam_intrinsics = data[:3,:3]
        return cam_intrinsics

    def rescale_intrinsics(self, K, img_hw_orig, img_hw_new):
        K[0,:] = K[0,:] * img_hw_new[1] / img_hw_orig[1]
        K[1,:] = K[1,:] * img_hw_new[0] / img_hw_orig[0]
        return K

    def get_intrinsics_per_scale(self, K, scale):
        K_new = copy.deepcopy(K)
        K_new[0,:] = K_new[0,:] / (2**scale)
        K_new[1,:] = K_new[1,:] / (2**scale)
        K_new_inv = np.linalg.inv(K_new)
        return K_new, K_new_inv

    def get_multiscale_intrinsics(self, K, num_scales):
        K_ms, K_inv_ms = [], []
        for s in range(num_scales):
            K_new, K_new_inv = self.get_intrinsics_per_scale(K, s)
            K_ms.append(K_new[None,:,:])
            K_inv_ms.append(K_new_inv[None,:,:])
        K_ms = np.concatenate(K_ms, 0)
        K_inv_ms = np.concatenate(K_inv_ms, 0)
        return K_ms, K_inv_ms

    def __getitem__(self, idx):
        '''
        Returns:
        - img		torch.Tensor (N * H, W, 3)
        - K	torch.Tensor (num_scales, 3, 3)
        - K_inv	torch.Tensor (num_scales, 3, 3)
        '''
        if self.num_iterations is not None:
            idx = self.rand_num(idx)
        data = self.data_list[idx]
        # load img
        img = cv2.imread(data['image_file'])
        img_hw_orig = (int(img.shape[0] / 2), img.shape[1])
        img = self.preprocess_img(img, self.img_hw) # (img_h * 2, img_w, 3)
        img = img.transpose(2,0,1)

        # load intrinsic
        cam_intrinsic = self.read_cam_intrinsic(data['cam_intrinsic_file'])
        cam_intrinsic = self.rescale_intrinsics(cam_intrinsic, img_hw_orig, self.img_hw)
        K_ms, K_inv_ms = self.get_multiscale_intrinsics(cam_intrinsic, self.num_scales) # (num_scales, 3, 3), (num_scales, 3, 3)
        return torch.from_numpy(img).float(), torch.from_numpy(K_ms).float(), torch.from_numpy(K_inv_ms).float()

if __name__ == '__main__':
    pass

