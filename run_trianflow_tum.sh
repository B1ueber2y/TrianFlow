# TUM_tracks='Handheld/rgbd_dataset_freiburg1_360 TestingDebugging/rgbd_dataset_freiburg1_rpy StructureTexture/rgbd_dataset_freiburg3_nostructure_notexture_far StructureTexture/rgbd_dataset_freiburg3_nostructure_texture_near_withloop'

TUM_tracks='Handheld/rgbd_dataset_freiburg1_360'


for track in $TUM_tracks
do

    python infer_vo.py  --config_file ./config/tum.yaml \
                        --gpu 0 \
                        --traj_save_dir_txt results/tum/$track/trianflow_results.txt \
                        --sequences_root_dir /media/airlabsimulation/Acer/Olaya_data/Datasets/SLAM/tum_rgbd\
                        --sequence $track \
                        --pretrained_model /media/airlabsimulation/Acer/Olaya_data/SLAM_SOA/learning-based/TrianFlow/models/tum.pth 
                
done   

# for track in $TUM_tracks
# do
#     python ./core/evaluation/eval_odom.py   --dataset euroc \
#                                             --gt_txt /media/airlabsimulation/Acer/Olaya_data/Datasets/SLAM/euroc/$track.txt/mav0/state_groundtruth_estimate0/data.csv \
#                                             --result_txt results/euroc/$track/trianflow_results.txt \
#                                             --seq $track
                                            
# done