KITTI_tracks='00 01 02 03 04 05 06 07 08 09 10'

for track in $KITTI_tracks
do

    python infer_vo.py  --config_file ./config/odo.yaml \
                        --gpu 0 \
                        --traj_save_dir_txt results/kitti/$track/trianflow_results.txt \
                        --sequences_root_dir /media/airlabsimulation/Acer/Olaya_data/Datasets/SLAM/KITTI/data_odometry_color/dataset/sequences \
                        --sequence $track \
                        --pretrained_model /media/airlabsimulation/Acer/Olaya_data/SLAM_SOA/TrianFlow/models/kitti_odo.pth 
                
done   

# for track in $KITTI_tracks
# do
#     python ./core/evaluation/eval_odom.py   --dataset kitti --gt_txt /media/airlabsimulation/Acer/Olaya_data/Datasets/SLAM/KITTI/data_odometry_poses/dataset/poses/$track.txt \
#                                             --result_txt results/kitti/$track/trianflow_results.txt \
#                                             --seq $track
# done