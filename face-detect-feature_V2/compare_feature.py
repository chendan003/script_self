# -*-coding: utf-8 -*-
from matio import  load_mat
import numpy as np
from numpy.linalg import norm
import os.path as osp
import operator
import math
import json
from draw_cropped_face import *
import sys
import os
import cv2
import urllib

def load_feat(feat_file, flatten=True):
    feat = None
    if feat_file.endswith('npy'):
        feat = load_npy(feat_file)
    elif feat_file.endswith('bin'):
        feat = load_mat(feat_file)
    else:
        raise Exception(
            'Unsupported feature file. Only support .npy and .bin (OpenCV Mat file)')
    if flatten:
        feat = feat.flatten()
    return feat

def load_npy(npy_file):
    mat = None
    if osp.exists(npy_file):
        mat = np.load(npy_file)
    else:
        err_info = 'Can not find file: ' + npy_file
        raise Exception(err_info)

    return mat

def calc_similarity_cosine(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)
    sim = np.dot(feat1, feat2.T) / (feat1_norm * feat2_norm)
    return sim

def get_npylist(path):
    if osp.exists(path):
	all_list = os.listdir(path)
	npy_list = [ii for ii in all_list if ii.endswith('npy')]
    return npy_list

def download_images(url_list_file, save_src_img_dir):
    if not osp.exists(save_src_img_dir):
        os.mkdir(save_src_img_dir)

    with open(url_list_file, 'r') as fu:
        for url in fu:
	    url = url.strip()
            dst_name = osp.join(save_src_img_dir, osp.basename(url))
            urllib.urlretrieve(url, dst_name)

def draw_two_face_together(img1, img2, sim, dst_cropimg_dir):
    im1 = cv2.imread(img1)
    im2 = cv2.imread(img2)
# resize 256 x 256 and combine two images 
    im1_new = cv2.resize(im1, (256,320),interpolation=cv2.INTER_AREA)
    im2_new = cv2.resize(im2, (256,320),interpolation=cv2.INTER_AREA)
    im = np.zeros((320,512, 3))
    im[:,:256,:] = im1_new
    im[:,256:,:] = im2_new
    save_name = "{0} vs {1}".format(osp.basename(img1), osp.basename(img2))
    print 'save_name:', save_name

    save_name_full = osp.join(dst_cropimg_dir, save_name)
    print 'save_name_full:', save_name_full

    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, str(round(sim,4)), (180,150),font,1.5,(0,0,255),4)
    cv2.imwrite(save_name_full, im)

def  out_put_det_feat_result(project_prefix_fn, src_img_dir,
               url_list_file,dst_cropimg_dir, save_similar_file):

    diku_img_prefix='idcard'
    scene_img_prefix='scene'

#crop rec and save extend image
    download_images(url_list_file, src_img_dir) # get images down
    draw_extend_cropped_image(project_prefix_fn, src_img_dir, dst_cropimg_dir)
    
# write similarity of diku and test images
    with open(save_similar_file, 'w')as log:
        npylist = get_npylist(project_prefix_fn)

	diku = [jj for jj in npylist if jj.startswith('idcard')]
        scene = [tt for tt in npylist if tt.startswith('scene')]
   	for dd in diku:
            dd_full = osp.join(project_prefix_fn, dd)
	    feat1 = load_feat(dd_full)
	    for uu in scene:
		uu_full =  osp.join(project_prefix_fn, uu)
		feat2 = load_feat(uu_full)
		sim =  calc_similarity_cosine(feat1, feat2)
		print 'sim is', sim
		img_name_dd = dd.replace('npy','jpg')
		img_name_uu = uu.replace('npy','jpg')
		img1 = osp.join(dst_cropimg_dir, img_name_dd)
		img2 = osp.join(dst_cropimg_dir, img_name_uu)
		if not osp.exists(img1):
		    print 'the image we want to merge does not exist!!!!'
		draw_two_face_together(img1, img2, sim, dst_cropimg_dir)

     		log.writelines(img_name_dd + '  ' + img_name_uu + '   '+ str(sim))
		log.writelines('\n')

if __name__ == "__main__":
    
    project_prefix_fn = sys.argv[1]
    src_img_dir = sys.argv[2]

    if len(sys.argv)==3:
	dst_cropimg_dir = 'cropped_image'
	save_similar_file = 'similarity_result.txt'
    else:
        dst_cropimg_dir = sys.argv[3]
        save_similar_file = sys.argv[4]
# deal diku image and scene image together
    out_put_det_feat_result(project_prefix_fn, src_img_dir,
                            dst_cropimg_dir, save_similar_file)
