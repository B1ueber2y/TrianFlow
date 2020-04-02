import os, sys
import numpy as np
import cv2
import pdb
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm


colorlib = [(0,0,255),(255,0,0),(0,255,0),(255,255,0),(0,255,255),(255,0,255),(0,0,0),(255,255,255)]

class Visualizer(object):
    def __init__(self, loss_weights_dict, dump_dir=None):
        self.loss_weights_dict = loss_weights_dict
        self.use_flow_error = (self.loss_weights_dict['flow_error'] > 0)
        self.dump_dir = dump_dir

        self.log_list = []

    def add_log_pack(self, log_pack):
        self.log_list.append(log_pack)

    def dump_log(self, fname=None):
        if fname is None:
            fname = self.dump_dir
        with open(fname, 'wb') as f:
            pickle.dump(self.log_list, f)

    def print_loss(self, loss_pack, iter_=None):
        loss_pixel = loss_pack['loss_pixel'].mean().detach().cpu().numpy()
        loss_ssim = loss_pack['loss_ssim'].mean().detach().cpu().numpy()
        loss_flow_smooth = loss_pack['loss_flow_smooth'].mean().detach().cpu().numpy()
        loss_flow_consis = loss_pack['loss_flow_consis'].mean().detach().cpu().numpy()
        if 'pt_depth_loss' in loss_pack.keys():
            loss_pt_depth = loss_pack['pt_depth_loss'].mean().detach().cpu().numpy()
            loss_pj_depth = loss_pack['pj_depth_loss'].mean().detach().cpu().numpy()
            loss_depth_smooth = loss_pack['depth_smooth_loss'].mean().detach().cpu().numpy()
            str_= ('iter: {0}, loss_pixel: {1:.6f}, loss_ssim: {2:.6f}, loss_pt_depth: {3:.6f}, loss_pj_depth: {4:.6f}, loss_depth_smooth: {5:.6f}'.format(\
                iter_, loss_pixel, loss_ssim, loss_pt_depth, loss_pj_depth, loss_depth_smooth))
            if self.use_flow_error:
                loss_flow_error = loss_pack['flow_error'].mean().detach().cpu().numpy()
                str_ = str_ + ', loss_flow_error: {0:.6f}'.format(loss_flow_error)
            print(str_)
        else:
            print('iter: {4}, loss_pixel: {0:.6f}, loss_ssim: {1:.6f}, loss_flow_smooth: {2:.6f}, loss_flow_consis: {3:.6f}'.format(loss_pixel, loss_ssim, loss_flow_smooth, loss_flow_consis, iter_))

class Visualizer_debug():
    def __init__(self, dump_dir=None, img1=None, img2=None):
        self.dump_dir = dump_dir
        self.img1 = img1
        self.img2 = img2
    
    def draw_point_corres(self, batch_idx, match, name):
        img1 = self.img1[batch_idx]
        img2 = self.img2[batch_idx]
        self.show_corres(img1, img2, match, name)
        print("Correspondence Saved in " + self.dump_dir + '/' + name)

    def draw_invalid_corres_ray(self, img1, img2, depth_match, point2d_1_coord, point2d_2_coord, point2d_1_depth, point2d_2_depth, P1, P2):
        # img: [H, W, 3] match: [4, n] point2d_coord: [n, 2] P: [3, 4]
        idx = np.where(point2d_1_depth < 0)[0]
        select_match = depth_match[:, idx]
        self.show_corres(img1, img2, select_match)
        pdb.set_trace()
    
    def draw_epipolar_line(self, batch_idx, match, F, name):
        # img: [H, W, 3] match: [4,n] F: [3,3]
        img1 = self.img1[batch_idx]
        img2 = self.img2[batch_idx]
        self.show_epipolar_line(img1, img2, match, F, name)
        print("Epipolar Lines Saved in " + self.dump_dir + '/' + name)

    def show_corres(self, img1, img2, match, name):
        # img: [H, W, 3] match: [4, n]
        cv2.imwrite(os.path.join(self.dump_dir, name+'_img1_cor.png'), img1)
        cv2.imwrite(os.path.join(self.dump_dir, name+'_img2_cor.png'), img2)
        img1 = cv2.imread(os.path.join(self.dump_dir, name+'_img1_cor.png'))
        img2 = cv2.imread(os.path.join(self.dump_dir, name+'_img2_cor.png'))
        n = np.shape(match)[1]
        for i in range(n):
            x1,y1 = match[:2,i]
            x2,y2 = match[2:,i]
            #print((x1, y1))
            #print((x2, y2))
            cv2.circle(img1, (x1,y1), radius=1, color=colorlib[i%len(colorlib)], thickness=2)
            cv2.circle(img2, (x2,y2), radius=1, color=colorlib[i%len(colorlib)], thickness=2)
        cv2.imwrite(os.path.join(self.dump_dir, name+'_img1_cor.png'), img1)
        cv2.imwrite(os.path.join(self.dump_dir, name+'_img2_cor.png'), img2)
    
    def show_mask(self, mask, name):
        # mask: [H, W, 1]
        mask = mask / np.max(mask) * 255.0
        cv2.imwrite(os.path.join(self.dump_dir, name+'.png'), mask)
    
    def save_img(self, img, name):
        cv2.imwrite(os.path.join(self.dump_dir, name+'.png'), img)
    
    def save_depth_img(self, depth, name):
        # depth: [h,w,1]
        minddepth = np.min(depth)
        maxdepth = np.max(depth)
        depth_nor = (depth-minddepth) / (maxdepth-minddepth) * 255.0
        depth_nor = depth_nor.astype(np.uint8)
        cv2.imwrite(os.path.join(self.dump_dir, name+'_depth.png'), depth_nor)
    
    def save_disp_color_img(self, disp, name):
        vmax = np.percentile(disp, 95)
        normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp)[:,:,:3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        
        name_dest_im = os.path.join(self.dump_dir, name + '_depth.jpg')
        im.save(name_dest_im)


    def drawlines(self, img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c, _ = img1.shape
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),3,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),3,color,-1)
        return img1,img2
    
    def show_epipolar_line(self, img1, img2, match, F, name):
        # img: [H,W,3] match: [4,n] F: [3,3]
        pts1 = np.transpose(match[:2,:], [1,0])
        pts2 = np.transpose(match[2:,:], [1,0])
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = self.drawlines(img1,img2,lines1,pts1,pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = self.drawlines(img2,img1,lines2,pts2,pts1)

        cv2.imwrite(os.path.join(self.dump_dir, name+'_1eline.png'), img5)
        cv2.imwrite(os.path.join(self.dump_dir, name+'_2eline.png'), img3)

        return None

    
    def show_ray(self, ax, K, RT, point2d, cmap='Greens'):
        K_inv = np.linalg.inv(K)
        R, T = RT[:,:3], RT[:,3]
        ray_direction = np.matmul(np.matmul(R.T, K_inv), np.array([point2d[0], point2d[1], 1]))
        ray_direction = ray_direction / (np.linalg.norm(ray_direction, ord=2) + 1e-12)
        ray_origin = (-1) * np.matmul(R.T, T)

        scatters = [ray_origin + t * ray_direction for t in np.linspace(0.0, 100.0, 1000)]
        scatters = np.stack(scatters, axis=0)
        self.visualize_points(ax, scatters, cmap=cmap)
        self.scatter_3d(ax, scatters[0], scatter_color='r')
        return ray_direction

    def visualize_points(self, ax, points, cmap=None):
        # ax.plot3D(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap=cmap)
        # ax.plot3D(points[:,0], points[:,1], points[:,2], c=points[:,2])
        ax.plot3D(points[:,0], points[:,1], points[:,2])

    def scatter_3d(self, ax, point, scatter_color='r'):
        ax.scatter(point[0], point[1], point[2], c=scatter_color)

    def visualize_two_rays(self, ax, match, P1, P2):
        # match: [4] P: [3,4]
        K = P1[:,:3] # the first P1 has identity rotation matrix and zero translation.
        K_inv = np.linalg.inv(K)
        RT1, RT2 = np.matmul(K_inv, P1), np.matmul(K_inv, P2) 
        x1, y1, x2, y2 = match
        d1 = self.show_ray(ax, K, RT1, [x1, y1], cmap='Greens')
        d2 = self.show_ray(ax, K, RT2, [x2, y2], cmap='Reds')
        print(np.dot(d1.squeeze(), d2.squeeze()))
       
if __name__ == '__main__':
    img1 = cv2.imread('./vis/ga.png')
    img2 = cv2.imread('./vis/gb.png')
    match = np.load('./vis/gmatch.npy')
    print(np.shape(img1))
    match = np.reshape(match, [4,-1])
    select_match = match[:,np.random.randint(200000, size=100)]
    show_corres(img1, img2, select_match)
