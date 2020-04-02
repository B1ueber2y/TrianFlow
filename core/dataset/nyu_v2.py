import os, sys
import numpy as np
import imageio
import cv2
import copy
import h5py
import scipy.io as sio
import torch
import torch.utils.data
import pdb
from tqdm import tqdm
import torch.multiprocessing as mp

def collect_image_list(path):
    # Get ppm images list of a folder.
    files = os.listdir(path)
    sorted_file = sorted([f for f in files])
    image_list = []
    for l in sorted_file:
        if l.split('.')[-1] == 'ppm':
            image_list.append(l)
    return image_list

def process_folder(q, data_dir, output_dir, stride, train_scenes):
    # Directly process the original nyu v2 depth dataset.
    while True:
        if q.empty():
            break
        folder = q.get()
        scene_name = folder.split('/')[-1]
        s1,s2 = scene_name.split('_')[:-1], scene_name.split('_')[-1]
        scene_name_full = ''
        for j in s1:
            scene_name_full = scene_name_full + j + '_'
        scene_name_full = scene_name_full + s2[:4]
        
        if scene_name_full not in train_scenes:
            continue
        image_path = os.path.join(data_dir, folder)
        dump_image_path = os.path.join(output_dir, folder)
        if not os.path.isdir(dump_image_path):
            os.makedirs(dump_image_path)
        f = open(os.path.join(dump_image_path, 'train.txt'), 'w')
        
        # Note. the os.listdir method returns arbitary order of list. We need correct order.
        image_list = collect_image_list(image_path)
        #image_list = open(os.path.join(image_path, 'index.txt')).readlines()
        numbers = len(image_list) - 1  # The last ppm file seems truncated.
        for n in range(numbers - stride):
            s_idx = n
            e_idx = s_idx + stride
            s_name = image_list[s_idx].strip()
            e_name = image_list[e_idx].strip()
            
            curr_image = imageio.imread(os.path.join(image_path, s_name))
            next_image = imageio.imread(os.path.join(image_path, e_name))
            #curr_image = cv2.imread(os.path.join(image_path, s_name))
            #next_image = cv2.imread(os.path.join(image_path, e_name))
            seq_images = np.concatenate([curr_image, next_image], axis=0)
            imageio.imsave(os.path.join(dump_image_path,  os.path.splitext(s_name)[0]+'.png'), seq_images.astype('uint8'))
            #cv2.imwrite(os.path.join(dump_image_path, os.path.splitext(s_name)[0]+'.png'), seq_images.astype('uint8'))

            # Write training files
            #date = folder.split('_')[2]
            f.write('%s %s\n' % (os.path.join(folder, os.path.splitext(s_name)[0]+'.png'), 'calib_cam_to_cam.txt'))
        print(folder)

class NYU_Prepare(object):
    def __init__(self, data_dir, test_dir):
        self.data_dir = data_dir
        self.test_data = os.path.join(test_dir, 'nyu_depth_v2_labeled.mat')
        self.splits = os.path.join(test_dir, 'splits.mat')
        self.get_all_scenes()
        self.get_test_scenes()
        self.get_train_scenes()
        

    def __len__(self):
        raise NotImplementedError

    def get_all_scenes(self):
        self.all_scenes = []
        paths = os.listdir(self.data_dir)
        for p in paths:
            if os.path.isdir(os.path.join(self.data_dir, p)):
                pp = os.listdir(os.path.join(self.data_dir, p))
                for path in pp:
                    self.all_scenes.append(path)

    def get_test_scenes(self):
        self.test_scenes = []
        test_data = h5py.File(self.test_data, 'r')
        test_split = sio.loadmat(self.splits)['testNdxs']
        test_split = np.array(test_split).squeeze(1)
        
        test_scenes = test_data['scenes'][0][test_split-1]
        for i in range(len(test_scenes)):
            obj = test_data[test_scenes[i]]
            name = "".join(chr(j) for j in obj[:])
            if name not in self.test_scenes:
                self.test_scenes.append(name)
        #pdb.set_trace()
    
    def get_train_scenes(self):
        self.train_scenes = []
        train_data = h5py.File(self.test_data, 'r')
        train_split = sio.loadmat(self.splits)['trainNdxs']
        train_split = np.array(train_split).squeeze(1)
        
        train_scenes = train_data['scenes'][0][train_split-1]
        for i in range(len(train_scenes)):
            obj = train_data[train_scenes[i]]
            name = "".join(chr(j) for j in obj[:])
            if name not in self.train_scenes:
                self.train_scenes.append(name)


    def prepare_data_mp(self, output_dir, stride=1):
        num_processes = 32
        processes = []
        q = mp.Queue()
        if not os.path.isfile(os.path.join(output_dir, 'train.txt')):
            os.makedirs(output_dir)
            #f = open(os.path.join(output_dir, 'train.txt'), 'w')
            print('Preparing sequence data....')
            if not os.path.isdir(self.data_dir):
                raise
            dirlist = os.listdir(self.data_dir)
            total_dirlist = []
            # Get the different folders of images
            for d in dirlist:
                if not os.path.isdir(os.path.join(self.data_dir, d)):
                    continue
                seclist = os.listdir(os.path.join(self.data_dir, d))
                for s in seclist:
                    if os.path.isdir(os.path.join(self.data_dir, d, s)):
                        total_dirlist.append(os.path.join(d, s))
                        q.put(os.path.join(d, s))
            # Process every folder
            for rank in range(num_processes):
                p = mp.Process(target=process_folder, args=(q, self.data_dir, output_dir, stride, self.train_scenes))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        
        # Collect the training frames.
        f = open(os.path.join(output_dir, 'train.txt'), 'w')
        for dirlist in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, dirlist)):
                seclists = os.listdir(os.path.join(output_dir, dirlist))
                for s in seclists:
                    train_file = open(os.path.join(output_dir, dirlist, s, 'train.txt'), 'r')
                    for l in train_file.readlines():
                        f.write(l)
        f.close()
        
        f = open(os.path.join(output_dir, 'calib_cam_to_cam.txt'), 'w')
        f.write('P_rect: 5.1885790117450188e+02 0.0 3.2558244941119034e+02 0.0 0.0 5.1946961112127485e+02 2.5373616633400465e+02 0.0 0.0 0.0 1.0 0.0')
        f.close()
        print('Data Preparation Finished.')

    def __getitem__(self, idx):
        raise NotImplementedError



class NYU_v2(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_scales=3, img_hw=(448, 576), num_iterations=None):
        super(NYU_v2, self).__init__()
        self.data_dir = data_dir
        self.num_scales = num_scales
        self.img_hw = img_hw
        self.num_iterations = num_iterations
        self.undist_coeff = np.array([2.07966153e-01, -5.8613825e-01, 7.223136313e-04, 1.047962719e-03, 4.98569866e-01])
        self.mapx, self.mapy = None, None
        self.roi = None

        info_file = os.path.join(self.data_dir, 'train.txt')
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
    
    def undistort_img(self, img, K):
        img_h, img_w = img.shape[0], img.shape[1]
        img_hw_orig = (int(img_h / 2), img_w) 
        img1, img2 = img[:img_hw_orig[0], :, :], img[img_hw_orig[0]:, :, :]
        
        h, w = img_hw_orig
        if self.mapx is None:
            newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(K, self.undist_coeff, (w,h), 1, (w,h))
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(K, self.undist_coeff, None, newcameramtx, (w,h), 5)
        
        img1_undist = cv2.remap(img1, self.mapx, self.mapy, cv2.INTER_LINEAR)
        img2_undist = cv2.remap(img2, self.mapx, self.mapy, cv2.INTER_LINEAR)
        x,y,w,h = self.roi
        img1_undist = img1_undist[y:y+h, x:x+w]
        img2_undist = img2_undist[y:y+h, x:x+w]
        img_undist = np.concatenate([img1_undist, img2_undist], 0)
        #cv2.imwrite('./test.png', img)
        #cv2.imwrite('./test_undist.png', img_undist)
        #pdb.set_trace()
        return img_undist

    def preprocess_img(self, img, K, img_hw=None, is_test=False):
        if img_hw is None:
            img_hw = self.img_hw
        if not is_test:
            #img = img
            img = self.undistort_img(img, K)
            #img = self.random_flip_img(img)
            
        img = self.resize_img(img, img_hw)
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
        K_new = copy.deepcopy(K)
        K_new[0,:] = K_new[0,:] * img_hw_new[0] / img_hw_orig[0]
        K_new[1,:] = K_new[1,:] * img_hw_new[1] / img_hw_orig[1]
        return K_new

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
        if idx >= self.num_iterations:
            raise IndexError
        if self.num_iterations is not None:
            idx = self.rand_num(idx)
        data = self.data_list[idx]
        # load img
        img = cv2.imread(data['image_file'])
        img_hw_orig = (int(img.shape[0] / 2), img.shape[1])
        
        # load intrinsic
        cam_intrinsic_orig = self.read_cam_intrinsic(data['cam_intrinsic_file'])
        cam_intrinsic = self.rescale_intrinsics(cam_intrinsic_orig, img_hw_orig, self.img_hw)
        K_ms, K_inv_ms = self.get_multiscale_intrinsics(cam_intrinsic, self.num_scales) # (num_scales, 3, 3), (num_scales, 3, 3)
        
        # image preprocessing
        img = self.preprocess_img(img, cam_intrinsic_orig, self.img_hw) # (img_h * 2, img_w, 3)
        img = img.transpose(2,0,1)

        
        return torch.from_numpy(img).float(), torch.from_numpy(K_ms).float(), torch.from_numpy(K_inv_ms).float()





if __name__ == '__main__':
    data_dir = '/home4/zhaow/data/kitti'
    dirlist = os.listdir('/home4/zhaow/data/kitti')
    output_dir = '/home4/zhaow/data/kitti_seq/data_generated_s2'
    total_dirlist = []
    # Get the different folders of images
    for d in dirlist:
        seclist = os.listdir(os.path.join(data_dir, d))
        for s in seclist:
            if os.path.isdir(os.path.join(data_dir, d, s)):
                total_dirlist.append(os.path.join(d, s))
    
    F = open(os.path.join(output_dir, 'train.txt'), 'w')
    for p in total_dirlist:
        traintxt = os.path.join(os.path.join(output_dir, p), 'train.txt')
        f = open(traintxt, 'r')
        for line in f.readlines():
            F.write(line)
        print(traintxt)





