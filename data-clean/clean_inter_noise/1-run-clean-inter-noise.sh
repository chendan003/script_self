nohup python -u ./src/clean_inter_noise.py \
     --feature-root-folder=/workspace/data/blued_data/new_bad_blue_features0606 \
     --feature-list-path=/workspace/data/blued_data/bad_blue_features.txt \
     --feature-dims=512 \
     --threshold=0.7 \
     --noise-save-path=./0.7-bad_blue-result-clean-noise.txt \
     > ./logs/noise_clean.log 2>&1 &
 
