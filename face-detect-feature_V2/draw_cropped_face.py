# -*- coding:utf-8  -*-

import numpy as np
import cv2
import os
import os.path as osp
import json

def draw_extend_cropped_image(json_dir, img_dir, save_dir_file):
    fn = osp.join(json_dir, 'facex_feature_resp_list.json')
    feature_resp_list = open(fn, 'r')
    face = json.load(feature_resp_list)

    if not os.path.exists(save_dir_file):
        os.mkdir(save_dir_file)

    for uu in face:
        pts = uu['pts']
        url_name = uu['uri']
        feat_data = uu['feat_data']
        feat_basename = os.path.basename(feat_data)
#    save_name = os.path.basename(feat_basename.split('.')[0]+'.jpg')
        save_name = feat_basename[:-4] + '.jpg'
        save_path = os.path.join(save_dir_file, save_name)

        rec = [pts[0][0],pts[0][1],pts[2][0],pts[2][1]]
        ct_x = (rec[2] + rec[0])/2
        ct_y = (rec[3] + rec[1])/2
        w_rec = (rec[2] - rec[0])
        h_rec = (rec[3] - rec[1])

     
        rec_2 =  [ct_x -  w_rec, ct_y - h_rec, ct_x + w_rec, ct_y + h_rec]
        basename = os.path.basename(url_name)
        img_full = os.path.join(img_dir, basename)
        if not os.path.exists(img_full):
            print 'cann\'t find image!', img_full
        
	image = cv2.imread(img_full)
        w_img = image.shape[1]
        h_img = image.shape[0]
        new_img = np.zeros((2*h_rec+1, 2*w_rec+1, 3))
# rec in origial image
        x1_ = max(0, rec_2[0])
        x2_ = min(w_img, rec_2[2])
        y1_ = max(0, rec_2[1])
        y2_ = min(h_img, rec_2[3])
        w_t = x2_-x1_
        h_t = y2_-y1_
        print 'h_t', h_t
        print 'w_t', w_t
# eadge 
        if rec_2[0] < 0:
            x1_tar = abs(rec_2[0])
        else:
            x1_tar = 0
        if rec_2[1] < 0:
            y1_tar = abs(rec_2[1])
        else:
            y1_tar = 0


        new_img[y1_tar:(y1_tar + h_t), x1_tar:(x1_tar + w_t),:] = image[y1_:y2_, x1_:x2_, :]
        print( new_img[y1_tar:(y1_tar + h_t), x1_tar:(x1_tar + w_t), :].shape)
        #new_img = image[y1_tar: y2_tar, x1_tar:x2_tar]
        print (image[y1_:y2_, x1_:x2_ ,:].shape)
        cv2.imwrite(save_path, new_img)

def draw_orgrec_cropped_image(json_dir, img_dir, save_dir_file):
    fn = osp.join(json_dir, 'facex_feature_resp_list.json')
    feature_resp_list = open(fn, 'r')
    face = json.load(feature_resp_list)

    if not os.path.exists(save_dir_file):
        os.mkdir(save_dir_file)   
    
    for uu in face:
        pts = uu['pts']
        url_name = uu['uri']
        feat_data = uu['feat_data']
        feat_basename = os.path.basename(feat_data)
        save_name = os.path.basename(feat_basename.split('.')[0]+'.jpg')
        save_path = os.path.join(save_dir_file, save_name)
        rec = [pts[0][0],pts[0][1],pts[2][0],pts[2][1]]
        basename = os.path.basename(url_name)
        img_full = os.path.join(img_dir, basename)
        if not imghdr.what(img_full):
            continue

        image = cv2.imread(img_full)
    #print image
        new_img = image.copy()
        rec_new = new_img[rec[1]: rec[3], rec[0]:rec[2]]
    #print rec_new
        cv2.imwrite(save_path, rec_new) 










