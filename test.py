import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset import KITTI_2012, KITTI_2015
from core.evaluation import eval_flow_avg, load_gt_flow_kitti
from core.evaluation import eval_depth
from core.visualize import Visualizer_debug
from core.networks import Model_depth_pose, Model_flow, Model_flowposenet
from core.evaluation import load_gt_flow_kitti, load_gt_mask
import torch
from tqdm import tqdm
import pdb
import cv2
import numpy as np
import yaml

def test_kitti_2012(cfg, model, gt_flows, noc_masks):
    dataset = KITTI_2012(cfg.gt_2012_dir)
    flow_list = []
    for idx, inputs in enumerate(tqdm(dataset)):
        img, K, K_inv = inputs
        img = img[None,:,:,:]
        K = K[None,:,:]
        K_inv = K_inv[None,:,:]
        img_h = int(img.shape[2] / 2)
        img1, img2 = img[:,:,:img_h,:], img[:,:,img_h:,:]
        img1, img2, K, K_inv = img1.cuda(), img2.cuda(), K.cuda(), K_inv.cuda()
        if cfg.mode == 'flow' or cfg.mode == 'flowposenet':
            flow = model.inference_flow(img1, img2)
        else:
            flow, _, _, _, _, _ = model.inference(img1, img2, K, K_inv)
        #pdb.set_trace()
        flow = flow[0].detach().cpu().numpy()
        flow = flow.transpose(1,2,0)
        flow_list.append(flow)
        
    eval_flow_res = eval_flow_avg(gt_flows, noc_masks, flow_list, cfg, write_img=False)
    
    print('CONFIG: {0}, mode: {1}'.format(cfg.config_file, cfg.mode))
    print('[EVAL] [KITTI 2012]')
    print(eval_flow_res)
    return eval_flow_res

def test_kitti_2015(cfg, model, gt_flows, noc_masks, gt_masks, depth_save_dir=None):
    dataset = KITTI_2015(cfg.gt_2015_dir)
    visualizer = Visualizer_debug(depth_save_dir)
    pred_flow_list = []
    pred_disp_list = []
    img_list = []
    for idx, inputs in enumerate(tqdm(dataset)):
        img, K, K_inv = inputs
        img = img[None,:,:,:]
        K = K[None,:,:]
        K_inv = K_inv[None,:,:]
        img_h = int(img.shape[2] / 2)
        img1, img2 = img[:,:,:img_h,:], img[:,:,img_h:,:]
        img_list.append(img1)
        img1, img2, K, K_inv = img1.cuda(), img2.cuda(), K.cuda(), K_inv.cuda()
        if cfg.mode == 'flow' or cfg.mode == 'flowposenet':
            flow = model.inference_flow(img1, img2)
        else:
            flow, disp1, disp2, Rt, _, _ = model.inference(img1, img2, K, K_inv)
            disp = disp1[0].detach().cpu().numpy()
            disp = disp.transpose(1,2,0)
            pred_disp_list.append(disp)

        flow = flow[0].detach().cpu().numpy()
        flow = flow.transpose(1,2,0)
        pred_flow_list.append(flow)
        
    #pdb.set_trace()
    eval_flow_res = eval_flow_avg(gt_flows, noc_masks, pred_flow_list, cfg, moving_masks=gt_masks, write_img=False)
    print('CONFIG: {0}, mode: {1}'.format(cfg.config_file, cfg.mode))
    print('[EVAL] [KITTI 2015]')
    print(eval_flow_res)
    ## depth evaluation
    return eval_flow_res

def disp2depth(disp, min_depth=0.001, max_depth=80.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def resize_depths(gt_depth_list, pred_disp_list):
    gt_disp_list = []
    pred_depth_list = []
    pred_disp_resized = []
    for i in range(len(pred_disp_list)):
        h, w = gt_depth_list[i].shape
        pred_disp = cv2.resize(pred_disp_list[i], (w,h))
        pred_depth = 1.0 / (pred_disp + 1e-4)
        pred_depth_list.append(pred_depth)
        pred_disp_resized.append(pred_disp)
    
    return pred_depth_list, pred_disp_resized


def test_eigen_depth(cfg, model):
    print('Evaluate depth using eigen split. Using model in ' + cfg.model_dir)
    filenames = open('./data/eigen/test_files.txt').readlines()
    pred_disp_list = []
    for i in range(len(filenames)):
        path1, idx, _ = filenames[i].strip().split(' ')
        img = cv2.imread(os.path.join(os.path.join(cfg.raw_base_dir, path1), 'image_02/data/'+str(idx)+'.png'))
        #img_resize = cv2.resize(img, (832,256))
        img_resize = cv2.resize(img, (cfg.img_hw[1], cfg.img_hw[0]))
        img_input = torch.from_numpy(img_resize / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        disp = model.infer_depth(img_input)
        disp = disp[0].detach().cpu().numpy()
        disp = disp.transpose(1,2,0)
        pred_disp_list.append(disp)
        #print(i)
    
    gt_depths = np.load('./data/eigen/gt_depths.npz', allow_pickle=True)['data']
    pred_depths, pred_disp_resized = resize_depths(gt_depths, pred_disp_list)
    eval_depth_res = eval_depth(gt_depths, pred_depths)
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = eval_depth_res
    sys.stderr.write(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".
        format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                'a1', 'a2', 'a3'))
    sys.stderr.write(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".
        format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))

    return eval_depth_res


def resize_disp(pred_disp_list, gt_depths):
    pred_depths = []
    h, w = gt_depths[0].shape[0], gt_depths[0].shape[1]
    for i in range(len(pred_disp_list)):
        disp = pred_disp_list[i]
        resize_disp = cv2.resize(disp, (w,h))
        depth = 1.0 / resize_disp
        pred_depths.append(depth)
    
    return pred_depths

import h5py
import scipy.io as sio
def load_nyu_test_data(data_dir):
    data = h5py.File(os.path.join(data_dir, 'nyu_depth_v2_labeled.mat'), 'r')
    splits = sio.loadmat(os.path.join(data_dir, 'splits.mat'))
    test = np.array(splits['testNdxs']).squeeze(1)
    images = np.transpose(data['images'], [0,1,3,2])
    depths = np.transpose(data['depths'], [0,2,1])
    images = images[test-1]
    depths = depths[test-1]
    return images, depths

def test_nyu(cfg, model, test_images, test_gt_depths):
    leng = test_images.shape[0]
    print('Test nyu depth on '+str(leng)+' images. Using depth model in '+cfg.model_dir)
    pred_disp_list = []
    crop_imgs = []
    crop_gt_depths = []
    for i in range(leng):
        img = test_images[i]
        img_crop = img[:,45:472,41:602]
        crop_imgs.append(img_crop)
        gt_depth_crop = test_gt_depths[i][45:472,41:602]
        crop_gt_depths.append(gt_depth_crop)
        #img = np.transpose(cv2.resize(np.transpose(img_crop, [1,2,0]), (576,448)), [2,0,1])
        img = np.transpose(cv2.resize(np.transpose(img_crop, [1,2,0]), (cfg.img_hw[1],cfg.img_hw[0])), [2,0,1])
        img_t = torch.from_numpy(img).float().cuda().unsqueeze(0) / 255.0
        disp = model.infer_depth(img_t)
        disp = np.transpose(disp[0].cpu().detach().numpy(), [1,2,0])
        pred_disp_list.append(disp)
    
    pred_depths = resize_disp(pred_disp_list, crop_gt_depths)
    eval_depth_res = eval_depth(crop_gt_depths, pred_depths, nyu=True)
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = eval_depth_res
    sys.stderr.write(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".
        format('abs_rel', 'sq_rel', 'rms', 'log10', 
                'a1', 'a2', 'a3'))
    sys.stderr.write(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".
        format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))
    
    return eval_depth_res

def test_single_image(img_path, model, training_hw, save_dir='./'):
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]
    img_resized = cv2.resize(img, (training_hw[1], training_hw[0]))
    img_t = torch.from_numpy(np.transpose(img_resized, [2,0,1])).float().cuda().unsqueeze(0) / 255.0
    disp = model.infer_depth(img_t)
    disp = np.transpose(disp[0].cpu().detach().numpy(), [1,2,0])
    disp_resized = cv2.resize(disp, (w,h))
    depth = 1.0 / (1e-6 + disp_resized)

    visualizer = Visualizer_debug(dump_dir=save_dir)
    visualizer.save_disp_color_img(disp_resized, name='demo')
    print('Depth prediction saved in ' + save_dir)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="TrianFlow testing."
    )
    arg_parser.add_argument('-c', '--config_file', default=None, help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    arg_parser.add_argument('--mode', type=str, default='depth', help='mode for testing.')
    arg_parser.add_argument('--task', type=str, default='kitti_depth', help='To test on which task, kitti_depth or kitti_flow or nyuv2 or demo')
    arg_parser.add_argument('--image_path', type=str, default=None, help='Set this only when task==demo. Depth demo for single image.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None, help='directory for loading flow pretrained models')
    arg_parser.add_argument('--result_dir', type=str, default=None, help='directory for saving predictions')

    args = arg_parser.parse_args()
    if not os.path.exists(args.config_file):
        raise ValueError('config file not found.')
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['img_hw'] = (cfg['img_hw'][0], cfg['img_hw'][1])
    #cfg['log_dump_dir'] = os.path.join(args.model_dir, 'log.pkl')
    cfg['model_dir'] = args.result_dir

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

    if args.mode == 'flow':
        model = Model_flow(cfg_new)
    elif args.mode == 'depth' or args.mode == 'flow_3stage':
        model = Model_depth_pose(cfg_new)
    elif args.mode == 'flowposenet':
        model = Model_flowposenet(cfg_new)
    
    if args.task == 'demo':
        model = Model_depth_pose(cfg_new)

    model.cuda()
    weights = torch.load(args.pretrained_model)
    model.load_state_dict(weights['model_state_dict'])
    model.eval()
    print('Model Loaded.')

    if args.task == 'kitti_depth':
        depth_res = test_eigen_depth(cfg_new, model)
    elif args.task == 'kitti_flow':
        gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti(cfg_new.gt_2015_dir, 'kitti_2015')
        gt_masks_2015 = load_gt_mask(cfg_new.gt_2015_dir)
        flow_res = test_kitti_2015(cfg_new, model, gt_flows_2015, noc_masks_2015, gt_masks_2015)
    elif args.task == 'nyuv2':
        test_images, test_gt_depths = load_nyu_test_data(cfg_new.nyu_test_dir)
        depth_res = test_nyu(cfg_new, model, test_images, test_gt_depths)
    elif args.task == 'demo':
        test_single_image(args.image_path, model, training_hw=cfg['img_hw'], save_dir=args.result_dir)

