import os, sys
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset import KITTI_RAW, KITTI_Prepared, NYU_Prepare, NYU_v2, KITTI_Odo
from core.networks import get_model
from core.config import generate_loss_weights_dict
from core.visualize import Visualizer
from core.evaluation import load_gt_flow_kitti, load_gt_mask
from test import test_kitti_2012, test_kitti_2015, test_eigen_depth, test_nyu, load_nyu_test_data

from collections import OrderedDict
import torch
import torch.utils.data
from tqdm import tqdm
import shutil
import pickle
import pdb

def save_model(iter_, model_dir, filename, model, optimizer):
    torch.save({"iteration": iter_, "model_state_dict": model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_dir, filename))

def load_model(model_dir, filename, model, optimizer):
    data = torch.load(os.path.join(model_dir, filename))
    iter_ = data['iteration']
    model.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return iter_, model, optimizer

def train(cfg):
    # load model and optimizer
    model = get_model(cfg.mode)(cfg)
    if cfg.multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': cfg.lr}])

    # Load Pretrained Models
    if cfg.resume:
        if cfg.iter_start > 0:
            cfg.iter_start, model, optimizer = load_model(cfg.model_dir, 'iter_{}.pth'.format(cfg.iter_start), model, optimizer)
        else:
            cfg.iter_start, model, optimizer = load_model(cfg.model_dir, 'last.pth', model, optimizer)
    elif cfg.flow_pretrained_model:
        data = torch.load(cfg.flow_pretrained_model)['model_state_dict']
        renamed_dict = OrderedDict()
        for k, v in data.items():
            if cfg.multi_gpu:
                name = 'module.model_flow.' + k
            elif cfg.mode == 'flowposenet':
                name = 'model_flow.' + k
            else:
                name = 'model_pose.model_flow.' + k
            renamed_dict[name] = v
        missing_keys, unexp_keys = model.load_state_dict(renamed_dict, strict=False)
        print(missing_keys)
        print(unexp_keys)
        print('Load Flow Pretrained Model from ' + cfg.flow_pretrained_model)
    if cfg.depth_pretrained_model and not cfg.resume:
        data = torch.load(cfg.depth_pretrained_model)['model_state_dict']
        if cfg.multi_gpu:
            renamed_dict = OrderedDict()
            for k, v in data.items():
                name = 'module.' + k
                renamed_dict[name] = v
            missing_keys, unexp_keys = model.load_state_dict(renamed_dict, strict=False)
        else:
            missing_keys, unexp_keys = model.load_state_dict(data, strict=False)
        print(missing_keys)
        print('##############')
        print(unexp_keys)
        print('Load Depth Pretrained Model from ' + cfg.depth_pretrained_model)
   
    loss_weights_dict = generate_loss_weights_dict(cfg)
    visualizer = Visualizer(loss_weights_dict, cfg.log_dump_dir)

    # load dataset
    data_dir = os.path.join(cfg.prepared_base_dir, cfg.prepared_save_dir)
    if not os.path.exists(os.path.join(data_dir, 'train.txt')):
        if cfg.dataset == 'kitti_depth':
            kitti_raw_dataset = KITTI_RAW(cfg.raw_base_dir, cfg.static_frames_txt, cfg.test_scenes_txt)
            kitti_raw_dataset.prepare_data_mp(data_dir, stride=1)
        elif cfg.dataset == 'kitti_odo':
            kitti_raw_dataset = KITTI_Odo(cfg.raw_base_dir)
            kitti_raw_dataset.prepare_data_mp(data_dir, stride=1)
        elif cfg.dataset == 'nyuv2':
            nyu_raw_dataset = NYU_Prepare(cfg.raw_base_dir, cfg.nyu_test_dir)
            nyu_raw_dataset.prepare_data_mp(data_dir, stride=10)
        else:
            raise NotImplementedError
        
    if cfg.dataset == 'kitti_depth':
        dataset = KITTI_Prepared(data_dir, num_scales=cfg.num_scales, img_hw=cfg.img_hw, num_iterations=(cfg.num_iterations - cfg.iter_start) * cfg.batch_size)
    elif cfg.dataset == 'kitti_odo':
        dataset = KITTI_Prepared(data_dir, num_scales=cfg.num_scales, img_hw=cfg.img_hw, num_iterations=(cfg.num_iterations - cfg.iter_start) * cfg.batch_size)
    elif cfg.dataset == 'nyuv2':
        dataset = NYU_v2(data_dir, num_scales=cfg.num_scales, img_hw=cfg.img_hw, num_iterations=(cfg.num_iterations - cfg.iter_start) * cfg.batch_size)
    else:
        raise NotImplementedError
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    if cfg.dataset == 'kitti_depth' or cfg.dataset == 'kitti_odo':
        gt_flows_2012, noc_masks_2012 = load_gt_flow_kitti(cfg.gt_2012_dir, 'kitti_2012')
        gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti(cfg.gt_2015_dir, 'kitti_2015')
        gt_masks_2015 = load_gt_mask(cfg.gt_2015_dir)
    elif cfg.dataset == 'nyuv2':
        test_images, test_gt_depths = load_nyu_test_data(cfg.nyu_test_dir)

    # training
    print('starting iteration: {}.'.format(cfg.iter_start))
    for iter_, inputs in enumerate(tqdm(dataloader)):
        if (iter_ + 1) % cfg.test_interval == 0 and (not cfg.no_test):
            model.eval()
            if args.multi_gpu:
                model_eval = model.module
            else:
                model_eval = model
            if cfg.dataset == 'kitti_depth' or cfg.dataset == 'kitti_odo':
                if not (cfg.mode == 'depth' or cfg.mode == 'flowposenet'):
                    eval_2012_res = test_kitti_2012(cfg, model_eval, gt_flows_2012, noc_masks_2012)
                    eval_2015_res = test_kitti_2015(cfg, model_eval, gt_flows_2015, noc_masks_2015, gt_masks_2015, depth_save_dir=os.path.join(cfg.model_dir, 'results'))
                    visualizer.add_log_pack({'eval_2012_res': eval_2012_res, 'eval_2015_res': eval_2015_res})
            elif cfg.dataset == 'nyuv2':
                if not cfg.mode == 'flow':
                    eval_nyu_res = test_nyu(cfg, model_eval, test_images, test_gt_depths)
                    visualizer.add_log_pack({'eval_nyu_res': eval_nyu_res})
            visualizer.dump_log(os.path.join(cfg.model_dir, 'log.pkl'))
        model.train()
        iter_ = iter_ + cfg.iter_start
        optimizer.zero_grad()
        inputs = [k.cuda() for k in inputs]
        loss_pack = model(inputs)
        if iter_ % cfg.log_interval == 0:
            visualizer.print_loss(loss_pack, iter_=iter_)

        loss_list = []
        for key in list(loss_pack.keys()):
            loss_list.append((loss_weights_dict[key] * loss_pack[key].mean()).unsqueeze(0))
        loss = torch.cat(loss_list, 0).sum()
        loss.backward()
        optimizer.step()
        if (iter_ + 1) % cfg.save_interval == 0:
            save_model(iter_, cfg.model_dir, 'iter_{}.pth'.format(iter_), model, optimizer)
            save_model(iter_, cfg.model_dir, 'last.pth'.format(iter_), model, optimizer)
    
    if cfg.dataset == 'kitti_depth':
        if cfg.mode == 'depth' or cfg.mode == 'depth_pose':
            eval_depth_res = test_eigen_depth(cfg, model_eval)

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="TrianFlow training pipeline."
    )
    arg_parser.add_argument('-c', '--config_file', default=None, help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    arg_parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
    arg_parser.add_argument('--iter_start', type=int, default=0, help='starting iteration.')
    arg_parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    arg_parser.add_argument('--num_workers', type=int, default=6, help='number of workers.')
    arg_parser.add_argument('--log_interval', type=int, default=100, help='interval for printing loss.')
    arg_parser.add_argument('--test_interval', type=int, default=2000, help='interval for evaluation.')
    arg_parser.add_argument('--save_interval', type=int, default=2000, help='interval for saving models.')
    arg_parser.add_argument('--mode', type=str, default='flow', help='training mode.')
    arg_parser.add_argument('--model_dir', type=str, default=None, help='directory for saving models')
    arg_parser.add_argument('--prepared_save_dir', type=str, default='data_s1', help='directory name for generated training dataset')
    arg_parser.add_argument('--flow_pretrained_model', type=str, default=None, help='directory for loading flow pretrained models')
    arg_parser.add_argument('--depth_pretrained_model', type=str, default=None, help='directory for loading depth pretrained models')
    arg_parser.add_argument('--resume', action='store_true', help='to resume training.')
    arg_parser.add_argument('--multi_gpu', action='store_true', help='to use multiple gpu for training.')
    arg_parser.add_argument('--no_test', action='store_true', help='without evaluation.')
    args = arg_parser.parse_args()
        #args.config_file = 'config/debug.yaml'
    if args.config_file is None:
        raise ValueError('config file needed. -c --config_file.')

    # set model
    if args.model_dir is None:
        args.model_dir = os.path.join('models', os.path.splitext(os.path.split(args.config_file)[1])[0])
    args.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_dir, args.mode)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.config_file):
        raise ValueError('config file not found.')
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['img_hw'] = (cfg['img_hw'][0], cfg['img_hw'][1])
    cfg['log_dump_dir'] = os.path.join(args.model_dir, 'log.pkl')
    shutil.copy(args.config_file, args.model_dir)

    # copy attr into cfg
    for attr in dir(args):
        if attr[:2] != '__':
            cfg[attr] = getattr(args, attr)

    # set gpu
    num_gpus = len(args.gpu.split(','))
    if (args.multi_gpu and num_gpus <= 1) or ((not args.multi_gpu) and num_gpus > 1):
        raise ValueError('Error! the number of gpus used in the --gpu argument does not match the argument --multi_gpu.')
    if args.multi_gpu:
        cfg['batch_size'] = cfg['batch_size'] * num_gpus
        cfg['num_iterations'] = int(cfg['num_iterations'] / num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    class pObject(object):
        def __init__(self):
            pass
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])
    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(cfg_new, f)

    # main function 
    train(cfg_new)

