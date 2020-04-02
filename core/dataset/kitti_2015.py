import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kitti_2012 import KITTI_2012

class KITTI_2015(KITTI_2012):
    def __init__(self, data_dir, img_hw=(256, 832)):
        super(KITTI_2015, self).__init__(data_dir, img_hw, init=False)
        self.num_total = 200

        self.data_list = self.get_data_list()

if __name__ == '__main__':
    pass

