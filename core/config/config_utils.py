import os, sys

def generate_loss_weights_dict(cfg):
    weight_dict = {}
    weight_dict['loss_pixel'] = 1 - cfg.w_ssim
    weight_dict['loss_ssim'] = cfg.w_ssim
    weight_dict['loss_flow_smooth'] = cfg.w_flow_smooth
    weight_dict['loss_flow_consis'] = cfg.w_flow_consis
    weight_dict['geo_loss'] = cfg.w_geo
    weight_dict['pt_depth_loss'] = cfg.w_pt_depth
    weight_dict['pj_depth_loss'] = cfg.w_pj_depth
    weight_dict['depth_smooth_loss'] = cfg.w_depth_smooth
    weight_dict['flow_error'] = cfg.w_flow_error
    return weight_dict

