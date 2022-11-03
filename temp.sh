# EUROC_tracks='MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult'
EUROC_tracks='MH_01_easy'

# for track in $EUROC_tracks
# do

#     python infer_vo.py  --config_file ./config/euroc.yaml \
#                         --gpu 0 \
#                         --traj_save_dir_txt results/euroc/$track/trianflow_results.txt \
#                         --sequences_root_dir /media/airlabsimulation/Acer/Olaya_data/Datasets/SLAM/euroc \
#                         --sequence $track \
#                         --pretrained_model /media/airlabsimulation/Acer/Olaya_data/SLAM_SOA/learning-based/TrianFlow/models/tum.pth 
                
# done   

for track in $EUROC_tracks
do
    python ./core/evaluation/eval_odom.py   --dataset euroc \
                                            --gt_txt /media/airlabsimulation/Acer/Olaya_data/Datasets/SLAM/euroc/$track/mav0/state_groundtruth_estimate0/data.csv \
                                            --result_txt results/euroc/$track/trianflow_results.txt \
                                            --seq $track
                                            
done