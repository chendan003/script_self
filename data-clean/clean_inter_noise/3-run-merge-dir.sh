nohup python -u ./src/movedir.py \
     --img-root-folder=/workspace/data/blued_data/clean_blue_data_good \
#     --inter-same-label-path=0.7-ms1m-result-clean-merge-label.txt \
    --inter-same-label-path=/workspace/data/blued_code/clean_inter_noise/temp.txt \
     > ./nohup_move.log 2>&1 &
 
