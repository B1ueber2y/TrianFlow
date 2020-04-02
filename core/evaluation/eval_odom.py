import copy
from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob
import pdb


def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y)/np.sum(X ** 2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


class KittiEvalOdom():
    # ----------------------------------------------------------------------
	# poses: N,4,4
	# pose: 4,4
	# ----------------------------------------------------------------------
    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def loadPoses(self, file_name):
        # ----------------------------------------------------------------------
		# Each line in the file should follow one of the following structures
		# (1) idx pose(3x4 matrix in terms of 12 numbers)
		# (2) pose(3x4 matrix in terms of 12 numbers)
		# ----------------------------------------------------------------------
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ")]
            withIdx = int(len(line_split) == 13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectory_distances(self, poses):
        # ----------------------------------------------------------------------
		# poses: dictionary: [frame_idx: pose]
		# ----------------------------------------------------------------------
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]   
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))
        return dist

    def rotation_error(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5*(a+b+c-1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    def translation_error(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def last_frame_from_segment_length(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        err = []
        dist = self.trajectory_distances(poses_gt)
        self.step_size = 10

        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(dist, first_frame, len_)

                # ----------------------------------------------------------------------
				# Continue if sequence not long enough
				# ----------------------------------------------------------------------
                if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                    continue

                # ----------------------------------------------------------------------
				# compute rotational and translational errors
				# ----------------------------------------------------------------------
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # ----------------------------------------------------------------------
				# compute speed 
				# ----------------------------------------------------------------------
                num_frames = last_frame - first_frame + 1.0
                speed = len_/(0.1*num_frames)

                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err
        
    def save_sequence_errors(self, err, file_name):
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def compute_overall_err(self, seq_err):
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err

    def plotPath(self, seq, poses_gt, poses_result):
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num =-1

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            # for pose in poses_dict[key]:
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0,3],  pose[2,3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:,0],  pos_xz[:,1], label = key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_"+(seq)
        plt.savefig(self.plot_path_dir + "/" + png_title + ".pdf", bbox_inches='tight', pad_inches=0)
        # plt.show()

    def compute_segment_error(self, seq_errs):
        # ----------------------------------------------------------------------
		# This function calculates average errors for different segment.
		# ----------------------------------------------------------------------

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []
        # ----------------------------------------------------------------------
		# Get errors
		# ----------------------------------------------------------------------
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
        # ----------------------------------------------------------------------
		# Compute average
		# ----------------------------------------------------------------------
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def scale_optimization(self, gt, pred):
        """ Optimize scaling factor
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            new_pred (4x4 array dict): predicted poses after optimization
        """
        pred_updated = copy.deepcopy(pred)
        xyz_pred = []
        xyz_ref = []
        for i in pred:
            pose_pred = pred[i]
            pose_ref = gt[i]
            xyz_pred.append(pose_pred[:3, 3])
            xyz_ref.append(pose_ref[:3, 3])
        xyz_pred = np.asarray(xyz_pred)
        xyz_ref = np.asarray(xyz_ref)
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    def eval(self, gt_txt, result_txt, seq=None):
        # gt_dir: the directory of groundtruth poses txt
        # results_dir: the directory of predicted poses txt
        self.plot_path_dir = os.path.dirname(result_txt) + "/plot_path"
        if not os.path.exists(self.plot_path_dir):
            os.makedirs(self.plot_path_dir)
        
        self.gt_txt = gt_txt

        ave_t_errs = []
        ave_r_errs = []

        poses_result = self.loadPoses(result_txt)
        poses_gt = self.loadPoses(self.gt_txt)

        # Pose alignment to first frame
        idx_0 = sorted(list(poses_result.keys()))[0]
        pred_0 = poses_result[idx_0]
        gt_0 = poses_gt[idx_0]
        for cnt in poses_result:
            poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
            poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

        # get XYZ
        xyz_gt = []
        xyz_result = []
        for cnt in poses_result:
            xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
            xyz_result.append([poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
        xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
        xyz_result = np.asarray(xyz_result).transpose(1, 0)

        r, t, scale = umeyama_alignment(xyz_result, xyz_gt, True)

        align_transformation = np.eye(4)
        align_transformation[:3:, :3] = r
        align_transformation[:3, 3] = t
        
        for cnt in poses_result:
            poses_result[cnt][:3, 3] *= scale
            poses_result[cnt] = align_transformation @ poses_result[cnt]

        # ----------------------------------------------------------------------
        # compute sequence errors
        # ----------------------------------------------------------------------
        seq_err = self.calc_sequence_errors(poses_gt, poses_result)

        # ----------------------------------------------------------------------
        # Compute segment errors
        # ----------------------------------------------------------------------
        avg_segment_errs = self.compute_segment_error(seq_err)

        # ----------------------------------------------------------------------
        # compute overall error
        # ----------------------------------------------------------------------
        ave_t_err, ave_r_err = self.compute_overall_err(seq_err)
        print("Sequence: " + seq)
        print("Translational error (%): ", ave_t_err*100)
        print("Rotational error (deg/100m): ", ave_r_err/np.pi*180*100)
        ave_t_errs.append(ave_t_err)
        ave_r_errs.append(ave_r_err)

        # Plotting
        self.plotPath(seq, poses_gt, poses_result)    

        print("-------------------- For Copying ------------------------------")
        for i in range(len(ave_t_errs)):
            print("{0:.2f}".format(ave_t_errs[i]*100))
            print("{0:.2f}".format(ave_r_errs[i]/np.pi*180*100))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='KITTI evaluation')
    parser.add_argument('--gt_txt', type=str, required=True, help="Groundtruth directory")
    parser.add_argument('--result_txt', type=str, required=True, help="Result directory")
    parser.add_argument('--seq', type=str, help="sequences to be evaluated", default='09')
    args = parser.parse_args()

    eval_tool = KittiEvalOdom()
    eval_tool.eval(args.gt_txt, args.result_txt, seq=args.seq)
