import os, sys
import numpy as np
import imageio
from tqdm import tqdm
import torch.multiprocessing as mp
import pdb

def process_folder(q, static_frames, test_scenes, data_dir, output_dir, stride=1):
    while True:
        if q.empty():
            break
        folder = q.get()
        if folder in static_frames.keys():
            static_ids = static_frames[folder]
        else:
            static_ids = []
        scene = folder.split('/')[1]
        if scene[:-5] in test_scenes:
            continue
        image_path = os.path.join(data_dir, folder, 'image_02/data')
        dump_image_path = os.path.join(output_dir, folder)
        if not os.path.isdir(dump_image_path):
            os.makedirs(dump_image_path)
        f = open(os.path.join(dump_image_path, 'train.txt'), 'w')
        
        # Note. the os.listdir method returns arbitary order of list. We need correct order.
        numbers = len(os.listdir(image_path))
        for n in range(numbers - stride):
            s_idx = n
            e_idx = s_idx + stride
            if '%.10d'%s_idx in static_ids or '%.10d'%e_idx in static_ids:
                #print('%.10d'%s_idx)
                continue
            curr_image = imageio.imread(os.path.join(image_path, '%.10d'%s_idx)+'.png')
            next_image = imageio.imread(os.path.join(image_path, '%.10d'%e_idx)+'.png')
            seq_images = np.concatenate([curr_image, next_image], axis=0)
            imageio.imsave(os.path.join(dump_image_path, '%.10d'%s_idx)+'.png', seq_images.astype('uint8'))

            # Write training files
            date = folder.split('/')[0]
            f.write('%s %s\n' % (os.path.join(folder, '%.10d'%s_idx)+'.png', os.path.join(date, 'calib_cam_to_cam.txt')))
        print(folder)


class KITTI_RAW(object):
    def __init__(self, data_dir, static_frames_txt, test_scenes_txt):
        self.data_dir = data_dir
        self.static_frames_txt = static_frames_txt
        self.test_scenes_txt = test_scenes_txt

    def __len__(self):
        raise NotImplementedError

    def collect_static_frame(self):
        f = open(self.static_frames_txt)
        static_frames = {}
        for line in f.readlines():
            line = line.strip()
            date, drive, frame_id = line.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id))
            if os.path.join(date, drive) not in static_frames.keys():
                static_frames[os.path.join(date, drive)] = []
            static_frames[os.path.join(date, drive)].append(curr_fid)
        return static_frames
    
    def collect_test_scenes(self):
        f = open(self.test_scenes_txt)
        test_scenes = []
        for line in f.readlines():
            line = line.strip()
            test_scenes.append(line)
        return test_scenes

    def prepare_data_mp(self, output_dir, stride=1):
        num_processes = 16
        processes = []
        q = mp.Queue()
        static_frames = self.collect_static_frame()
        test_scenes = self.collect_test_scenes()
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
                seclist = os.listdir(os.path.join(self.data_dir, d))
                for s in seclist:
                    if os.path.isdir(os.path.join(self.data_dir, d, s)):
                        total_dirlist.append(os.path.join(d, s))
                        q.put(os.path.join(d, s))
            # Process every folder
            for rank in range(num_processes):
                p = mp.Process(target=process_folder, args=(q, static_frames, test_scenes, self.data_dir, output_dir, stride))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        
        # Collect the training frames.
        f = open(os.path.join(output_dir, 'train.txt'), 'w')
        for date in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, date)):
                drives = os.listdir(os.path.join(output_dir, date))
                for d in drives:
                    train_file = open(os.path.join(output_dir, date, d, 'train.txt'), 'r')
                    for l in train_file.readlines():
                        f.write(l)
        
        # Get calib files
        for date in os.listdir(self.data_dir):
            command = 'cp ' + os.path.join(self.data_dir, date, 'calib_cam_to_cam.txt') + ' ' + os.path.join(output_dir, date, 'calib_cam_to_cam.txt')
            os.system(command)
        
        print('Data Preparation Finished.')



    def prepare_data(self, output_dir):
        static_frames = self.collect_static_frame()
        test_scenes = self.collect_test_scenes()
        if not os.path.isfile(os.path.join(output_dir, 'train.txt')):
            os.makedirs(output_dir)
            f = open(os.path.join(output_dir, 'train.txt'), 'w')
            print('Preparing sequence data....')
            if not os.path.isdir(self.data_dir):
                raise
            dirlist = os.listdir(self.data_dir)
            total_dirlist = []
            # Get the different folders of images
            for d in dirlist:
                seclist = os.listdir(os.path.join(self.data_dir, d))
                for s in seclist:
                    if os.path.isdir(os.path.join(self.data_dir, d, s)):
                        total_dirlist.append(os.path.join(d, s))
            # Process every folder
            for folder in tqdm(total_dirlist):
                if folder in static_frames.keys():
                    static_ids = static_frames[folder]
                else:
                    static_ids = []
                scene = folder.split('/')[1]
                if scene in test_scenes:
                    continue
                image_path = os.path.join(self.data_dir, folder, 'image_02/data')
                dump_image_path = os.path.join(output_dir, folder)
                if not os.path.isdir(dump_image_path):
                    os.makedirs(dump_image_path)
                # Note. the os.listdir method returns arbitary order of list. We need correct order.
                numbers = len(os.listdir(image_path))
                for n in range(numbers - 1):
                    s_idx = n
                    e_idx = s_idx + 1
                    if '%.10d'%s_idx in static_ids or '%.10d'%e_idx in static_ids:
                        print('%.10d'%s_idx)
                        continue
                    curr_image = imageio.imread(os.path.join(image_path, '%.10d'%s_idx)+'.png')
                    next_image = imageio.imread(os.path.join(image_path, '%.10d'%e_idx)+'.png')
                    seq_images = np.concatenate([curr_image, next_image], axis=0)
                    imageio.imsave(os.path.join(dump_image_path, '%.10d'%s_idx)+'.png', seq_images.astype('uint8'))
    
                    # Write training files
                    date = folder.split('/')[0]
                    f.write('%s %s\n' % (os.path.join(folder, '%.10d'%s_idx)+'.png', os.path.join(date, 'calib_cam_to_cam.txt')))
                print(folder)
        
        # Get calib files
        for date in os.listdir(self.data_dir):
            command = 'cp ' + os.path.join(self.data_dir, date, 'calib_cam_to_cam.txt') + ' ' + os.path.join(output_dir, date, 'calib_cam_to_cam.txt')
            os.system(command)
        
        return os.path.join(output_dir, 'train.txt')

    def __getitem__(self, idx):
        raise NotImplementedError


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





