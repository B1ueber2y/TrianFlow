AQUALOC_tracks='1 2 3 4 5 6'

for track in $AQUALOC_tracks
do

    python infer_vo.py  --config_file ./config/aqualoc.yaml \
                        --gpu 0 \
                        --traj_save_dir_txt results/aqualoc/Archaeological_site_sequences/archaeo_sequence_$track\_raw_data/trianflow_results.txt \
                        --sequences_root_dir /home/olaya/Datasets/Aqualoc/ \
                        --sequence Archaeological_site_sequences/archaeo_sequence_$track\_raw_data/raw_data/images_sequence_$track \
                        --pretrained_model /home/olaya/dev/slam-survey/TrianFlow/models/tum.pth 
                
done   

# for track in $AQUALOC_tracks
# do
#     python ./core/evaluation/eval_odom.py   --dataset aqualoc \
#                                             --gt_txt /media/airlabsimulation/Acer/Olaya_data/Datasets/SLAM/aqualoc/$track.txt/mav0/state_groundtruth_estimate0/data.csv \
#                                             --result_txt results/aqualoc/$track/trianflow_results.txt \
#                                             --seq $track
      
                                            
# done