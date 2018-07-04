# -*-coding: utf-8 -*-
from matio import  load_mat
import numpy as np
from numpy.linalg import norm
import os.path as osp
import operator
import math
import json
from draw_croped_face import draw_extend_cropped_image, draw_orgrec_cropped_image
import sys
import os


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
#    feat1_norm = math.sqrt(sum([i*i for i in feat1]))
#    feat2_norm = math.sqrt(sum([i*i for i in feat2]))
    #mul = sum([feat1[u]*feat2[u] for u in range(512)])
    #sim = mul/(feat1_norm*feat2_norm)
    return sim

def get_npylist(path):
    if osp.exists(path):
	all_list = os.listdir(path)
	npy_list = [ii for ii in all_list if ii.endswith('npy')]
    return npy_list

def  out_put_det_feat_result(project_prefix_fn, src_img_dir, dst_cropimg_dir, save_similar_file, diku_img_prefix='diku', scene_img_prefix='scene'):
    #facex_feature_resp_list = osp.join(project_prefix_fn, 'facex_feature_resp_list.json')
    #fn = open(facex_feature_resp_list, 'r')
    #ffrl = json.load(fn)

#crop rec and save extend image

    draw_extend_cropped_image(project_prefix_fn, src_img_dir, dst_cropimg_dir)
    
# write similarity of diku and test images
    with open(save_similar_file, 'w')as log:
        npylist = get_npylist(project_prefix_fn)
	print 'npylist is ', npylist
	diku = [jj for jj in npylist if jj.startswith('diku')]
        scene = [tt for tt in npylist if tt.startswith('scene')]
   	for dd in diku:
            dd_full = osp.join(project_prefix_fn, dd)
	    feat1 = load_feat(dd_full)
	    for uu in scene:
		uu_full =  osp.join(project_prefix_fn, uu)
		feat2 = load_feat(uu_full)
		sim =  calc_similarity_cosine(feat1, feat2)
		print 'sim is', sim
		print 'begin to write!'
     		log.writelines(dd.replace('npy','jpg') + '  ' + uu.replace('npy','jpg') + '   '+ str(sim))
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
    out_put_det_feat_result(project_prefix_fn, src_img_dir, dst_cropimg_dir, save_similar_file, diku_img_prefix='diku', scene_img_prefix='scene') 

