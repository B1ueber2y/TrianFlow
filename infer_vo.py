import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.networks.model_depth_pose import Model_depth_pose
from core.networks.model_flow import Model_flow
from visualizer import *
from profiler import Profiler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from sklearn import linear_model
import yaml
import warnings
import copy
from collections import OrderedDict
warnings.filterwarnings("ignore")

def save_traj(path, poses):
    """
    path: file path of saved poses
    poses: list of global poses
    """
    f = open(path, 'w')
    for i in range(len(poses)):
        pose = poses[i].flatten()[:12] # [3x4]
        line = " ".join([str(j) for j in pose])
        f.write(line + '\n')
    print('Trajectory Saved.')

def projection(xy, points, h_max, w_max):
    # Project the triangulation points to depth map. Directly correspondence mapping rather than projection.
    # xy: [N, 2] points: [3, N]
    depth = np.zeros((h_max, w_max))
    xy_int = np.around(xy).astype('int')

    # Ensure all the correspondences are inside the image.
    y_idx = (xy_int[:, 0] >= 0) * (xy_int[:, 0] < w_max)
    x_idx = (xy_int[:, 1] >= 0) * (xy_int[:, 1] < h_max)
    idx = y_idx * x_idx
    xy_int = xy_int[idx]
    points_valid = points[:, idx]

    depth[xy_int[:, 1], xy_int[:, 0]] = points_valid[2]
    return depth

def unprojection(xy, depth, K):
    # xy: [N, 2] image coordinates of match points
    # depth: [N] depth value of match points
    N = xy.shape[0]
    # initialize regular grid
    ones = np.ones((N, 1))
    xy_h = np.concatenate([xy, ones], axis=1)
    xy_h = np.transpose(xy_h, (1,0)) # [3, N]
    #depth = np.transpose(depth, (1,0)) # [1, N]
    
    K_inv = np.linalg.inv(K)
    points = np.matmul(K_inv, xy_h) * depth
    points = np.transpose(points) # [N, 3]
    return points

def cv_triangulation(matches, pose):
    # matches: [N, 4], the correspondence xy coordinates
    # pose: [4, 4], the relative pose trans from 1 to 2
    xy1 = matches[:, :2].transpose()
    xy2 = matches[:, 2:].transpose() # [2, N]
    pose1 = np.eye(4)
    pose2 = pose1 @ pose
    points = cv2.triangulatePoints(pose1[:3], pose2[:3], xy1, xy2)
    points /= points[3]

    points1 = pose1[:3] @ points
    points2 = pose2[:3] @ points
    return points1, points2

class infer_vo():
    def __init__(self, seq_id, sequences_root_dir):
        self.img_dir = sequences_root_dir
        #self.img_dir = '/home4/zhaow/data/kitti_odometry/sampled_s4_sequences/'
        self.seq_id = seq_id
        self.raw_img_h = 370.0#320
        self.raw_img_w = 1226.0#1024
        self.new_img_h = 256#320
        self.new_img_w = 832#1024
        self.max_depth = 50.0
        self.min_depth = 0.0
        self.cam_intrinsics = self.read_rescale_camera_intrinsics(os.path.join(self.img_dir, seq_id) + '/calib.txt')
        self.flow_pose_ransac_thre = 0.1 #0.2
        self.flow_pose_ransac_times = 10 #5
        self.flow_pose_min_flow = 5
        self.align_ransac_min_samples = 3
        self.align_ransac_max_trials = 100
        self.align_ransac_stop_prob = 0.99
        self.align_ransac_thre = 1.0
        self.PnP_ransac_iter = 1000
        self.PnP_ransac_thre = 1
        self.PnP_ransac_times = 5
    
    def read_rescale_camera_intrinsics(self, path):
        raw_img_h = self.raw_img_h
        raw_img_w = self.raw_img_w
        new_img_h = self.new_img_h
        new_img_w = self.new_img_w
        with open(path, 'r') as f:
            lines = f.readlines()
        data = lines[-1].strip('\n').split(' ')[1:]
        data = [float(k) for k in data]
        data = np.array(data).reshape(3,4)
        cam_intrinsics = data[:3,:3]
        cam_intrinsics[0,:] = cam_intrinsics[0,:] * new_img_w / raw_img_w
        cam_intrinsics[1,:] = cam_intrinsics[1,:] * new_img_h / raw_img_h
        return cam_intrinsics
    
    def load_images(self):
        path = self.img_dir
        seq = self.seq_id
        new_img_h = self.new_img_h
        new_img_w = self.new_img_w
        seq_dir = os.path.join(path, seq)
        image_dir = os.path.join(seq_dir, 'image_2')
        num = len(os.listdir(image_dir))
        images = []
        for i in range(num):
            image = cv2.imread(os.path.join(image_dir, '%.6d'%i)+'.png')
            image = cv2.resize(image, (new_img_w, new_img_h))
            images.append(image)
        return images
    
    def get_prediction(self, img1, img2, model, K, K_inv, match_num):
        # img1: [3,H,W] K: [3,3]
        #visualizer = Visualizer_debug('/home3/zhaow/TrianFlow-pytorch/vis/')
        img1_t = torch.from_numpy(np.transpose(img1 / 255.0, [2,0,1])).cuda().float().unsqueeze(0)
        img2_t = torch.from_numpy(np.transpose(img2 / 255.0, [2,0,1])).cuda().float().unsqueeze(0)
        K = torch.from_numpy(K).cuda().float().unsqueeze(0)
        K_inv = torch.from_numpy(K_inv).cuda().float().unsqueeze(0)

        filt_depth_match, depth1, depth2 = model.infer_vo(img1_t, img2_t, K, K_inv, match_num)
        return filt_depth_match[0].transpose(0,1).cpu().detach().numpy(), depth1[0].squeeze(0).cpu().detach().numpy(), depth2[0].squeeze(0).cpu().detach().numpy()

    
    def process_video(self, images, model):
        '''Process a sequence to get scale consistent trajectory results. 
        Register according to depth net predictions. Here we assume depth predictions have consistent scale.
        If not, pleas use process_video_tri which only use triangulated depth to get self-consistent scaled pose.
        '''
        poses = []
        global_pose = np.eye(4)
        # The first one global pose is origin.
        poses.append(copy.deepcopy(global_pose))
        seq_len = len(images)
        K = self.cam_intrinsics
        K_inv = np.linalg.inv(self.cam_intrinsics)
        for i in range(seq_len-1):
            img1, img2 = images[i], images[i+1]
            depth_match, depth1, depth2 = self.get_prediction(img1, img2, model, K, K_inv, match_num=5000)
            
            rel_pose = np.eye(4)
            flow_pose = self.solve_pose_flow(depth_match[:,:2], depth_match[:,2:])
            rel_pose[:3,:3] = copy.deepcopy(flow_pose[:3,:3])
            if np.linalg.norm(flow_pose[:3,3:]) != 0:
                scale = self.align_to_depth(depth_match[:,:2], depth_match[:,2:], flow_pose, depth2)
                rel_pose[:3,3:] = flow_pose[:3,3:] * scale
            
            if np.linalg.norm(flow_pose[:3,3:]) == 0 or scale == -1:
                print('PnP '+str(i))
                pnp_pose = self.solve_pose_pnp(depth_match[:,:2], depth_match[:,2:], depth1)
                rel_pose = pnp_pose

            global_pose[:3,3:] = np.matmul(global_pose[:3,:3], rel_pose[:3,3:]) + global_pose[:3,3:]
            global_pose[:3,:3] = np.matmul(global_pose[:3,:3], rel_pose[:3,:3])
            poses.append(copy.deepcopy(global_pose))
            print(i)
        return poses
    
    def normalize_coord(self, xy, K):
        xy_norm = copy.deepcopy(xy)
        xy_norm[:,0] = (xy[:,0] - K[0,2]) / K[0,0]
        xy_norm[:,1] = (xy[:,1] - K[1,2]) / K[1,1]

        return xy_norm
    
    def align_to_depth(self, xy1, xy2, pose, depth2):
        # Align the translation scale according to triangulation depth
        # xy1, xy2: [N, 2] pose: [4, 4] depth2: [H, W]
        
        # Triangulation
        img_h, img_w = np.shape(depth2)[0], np.shape(depth2)[1]
        pose_inv = np.linalg.inv(pose)

        xy1_norm = self.normalize_coord(xy1, self.cam_intrinsics)
        xy2_norm = self.normalize_coord(xy2, self.cam_intrinsics)

        points1_tri, points2_tri = cv_triangulation(np.concatenate([xy1_norm, xy2_norm], axis=1), pose_inv)
        
        depth2_tri = projection(xy2, points2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0
        
        # Remove negative depths
        valid_mask = (depth2 > 0) * (depth2_tri > 0)
        depth_pred_valid = depth2[valid_mask]
        depth_tri_valid = depth2_tri[valid_mask]
        
        if np.sum(valid_mask) > 100:
            scale_reg = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False), min_samples=self.align_ransac_min_samples, \
                        max_trials=self.align_ransac_max_trials, stop_probability=self.align_ransac_stop_prob, residual_threshold=self.align_ransac_thre)
            scale_reg.fit(depth_tri_valid.reshape(-1, 1), depth_pred_valid.reshape(-1, 1))
            scale = scale_reg.estimator_.coef_[0, 0]
        else:
            scale = -1

        return scale
    
    def solve_pose_pnp(self, xy1, xy2, depth1):
        # Use pnp to solve relative poses.
        # xy1, xy2: [N, 2] depth1: [H, W]

        img_h, img_w = np.shape(depth1)[0], np.shape(depth1)[1]
        
        # Ensure all the correspondences are inside the image.
        x_idx = (xy2[:, 0] >= 0) * (xy2[:, 0] < img_w)
        y_idx = (xy2[:, 1] >= 0) * (xy2[:, 1] < img_h)
        idx = y_idx * x_idx
        xy1 = xy1[idx]
        xy2 = xy2[idx]

        xy1_int = xy1.astype(np.int)
        sample_depth = depth1[xy1_int[:,1], xy1_int[:,0]]
        valid_depth_mask = (sample_depth < self.max_depth) * (sample_depth > self.min_depth)

        xy1 = xy1[valid_depth_mask]
        xy2 = xy2[valid_depth_mask]

        # Unproject to 3d space
        points1 = unprojection(xy1, sample_depth[valid_depth_mask], self.cam_intrinsics)

        # ransac
        best_rt = []
        max_inlier_num = 0
        max_ransac_iter = self.PnP_ransac_times
        
        for i in range(max_ransac_iter):
            if xy2.shape[0] > 4:
                flag, r, t, inlier = cv2.solvePnPRansac(objectPoints=points1, imagePoints=xy2, cameraMatrix=self.cam_intrinsics, distCoeffs=None, iterationsCount=self.PnP_ransac_iter, reprojectionError=self.PnP_ransac_thre)
                if flag and inlier.shape[0] > max_inlier_num:
                    best_rt = [r, t]
                    max_inlier_num = inlier.shape[0]
        pose = np.eye(4)
        if len(best_rt) != 0:
            r, t = best_rt
            pose[:3,:3] = cv2.Rodrigues(r)[0]
            pose[:3,3:] = t
        pose = np.linalg.inv(pose)
        return pose
    
    def solve_pose_flow(self, xy1, xy2):
        # Solve essential matrix to find relative pose from flow.

        # ransac
        best_rt = []
        max_inlier_num = 0
        max_ransac_iter = self.flow_pose_ransac_times
        best_inliers = np.ones((xy1.shape[0])) == 1
        pp = (self.cam_intrinsics[0,2], self.cam_intrinsics[1,2])
        
        # flow magnitude
        avg_flow = np.mean(np.linalg.norm(xy1 - xy2, axis=1))
        if avg_flow > self.flow_pose_min_flow:
            for i in range(max_ransac_iter):
                E, inliers = cv2.findEssentialMat(xy2, xy1, focal=self.cam_intrinsics[0,0], pp=pp, method=cv2.RANSAC, prob=0.99, threshold=self.flow_pose_ransac_thre)
                cheirality_cnt, R, t, _ = cv2.recoverPose(E, xy2, xy1, focal=self.cam_intrinsics[0,0], pp=pp)
                if inliers.sum() > max_inlier_num and cheirality_cnt > 50:
                    best_rt = [R, t]
                    max_inlier_num = inliers.sum()
                    best_inliers = inliers
            if len(best_rt) == 0:
                R = np.eye(3)
                t = np.zeros((3,1))
                best_rt = [R, t]
        else:
            R = np.eye(3)
            t = np.zeros((3,1))
            best_rt = [R, t]
        R, t = best_rt
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3:] = t
        return pose


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="TrianFlow training pipeline."
    )
    arg_parser.add_argument('-c', '--config_file', default=None, help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    arg_parser.add_argument('--mode', type=str, default='flow', help='training mode.')
    arg_parser.add_argument('--traj_save_dir_txt', type=str, default=None, help='directory for saving results')
    arg_parser.add_argument('--sequences_root_dir', type=str, default=None, help='directory for test sequences')
    arg_parser.add_argument('--sequence', type=str, default='09', help='Test sequence id.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None, help='directory for loading pretrained models')
    args = arg_parser.parse_args()

    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['dataset'] = 'kitti_odo'
    # copy attr into cfg
    for attr in dir(args):
        if attr[:2] != '__':
            cfg[attr] = getattr(args, attr)
    class pObject(object):
        def __init__(self):
            pass
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model = Model_depth_pose(cfg_new)
    model.cuda()
    weights = torch.load(args.pretrained_model)
    model.load_state_dict(weights['model_state_dict'])
    
    model.eval()
    print('Model Loaded.')

    print('Testing VO.')
    vo_test = infer_vo(args.sequence, args.sequences_root_dir)
    images = vo_test.load_images()
    print('Images Loaded. Total ' + str(len(images)) + ' images found.')
    poses = vo_test.process_video(images, model)
    print('Test completed.')

    traj_txt = args.traj_save_dir_txt
    save_traj(traj_txt, poses)
