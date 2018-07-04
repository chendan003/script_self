this script can compare the similarity of diku images and scene image by two steps

first step: detect face and extract features 
      change input_file
      exec the script facex_api_v3_request.py
      
second step: compare similarity between two faces.
      exec the script compare_feature.py 
      para1: the out put if first step file, in the example this para is "facex_api_response_wanrenli_#180704-081746"
      para2: original_images_dir
