import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_flow import Model_flow
from model_triangulate_pose import Model_triangulate_pose
from model_depth_pose import Model_depth_pose
from model_flowposenet import Model_flowposenet

def get_model(mode):
    if mode == 'flow':
        return Model_flow
    elif mode == 'pose' or mode == 'pose_flow':
        return Model_triangulate_pose
    elif mode == 'depth' or mode == 'depth_pose':
        return Model_depth_pose
    elif mode == 'flowposenet':
        return Model_flowposenet
    else:
        raise ValueError('Mode {} not found.'.format(mode))
