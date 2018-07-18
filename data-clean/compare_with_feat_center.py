# -*-coding:utf-8  -*-

import numpy as np
from matio import load_mat
import os.path as osp
from numpy.linalg import norm
import os

#choose image lfw from testdata (mixed)
def load_feat(feat_file, flatten=True):
    feat = None
    if feat_file.endswith('npy'):
        feat = np.load(feat_file)
    elif feat_file.endswith('bin'):
        feat = load_mat(feat_file)
    else:
        raise Exception(
            'Unsupported feature file. Only support .npy and .bin (OpenCV Mat file)')
    if flatten:
        feat = feat.flatten()
    return feat

def calc_similarity_cosine(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)
    sim = np.dot(feat1, feat2.T) / (feat1_norm * feat2_norm)
    return sim

def get_sonfile_list(path):
    if osp.exists(path) and osp.isdir(path):
        imglist = os.listdir(path)
#    else:
#       imglist = ''
#       print 'path does not exist!'
    return imglist


def mean_feature(path):
    sonfile = [ii for ii in get_sonfile_list(path) if ii.endswith('bin')]
    num = len(sonfile)
    sum_feature = load_feat(osp.join(path , sonfile[0]))

    for tt in sonfile[1:]:
        sum_feature +=load_feat(osp.join(path, tt))
    mean_feat = sum_feature/(num+0.00)
    return mean_feat

if __name__=='__main__':
    lfw_feature_dir = '/workspace/data/lfw-features/insightface-r100-spa-m2.0-ep96'
    lfw_mean = []
    sonfile_lfw = [uu for uu in get_sonfile_list(lfw_feature_dir)]
    print 'sonfile_lfw len is :', len(sonfile_lfw)
    for label in sonfile_lfw:
        full_path = osp.join(lfw_feature_dir, label)
        mean_feat = mean_feature(full_path)
        print 'temp label is:', full_path
        lfw_mean.append(mean_feat)

    usefull_fn = open('usefull_result/usefull_img27.txt', 'w')
    #useless_fn = open('useless_img.txt', 'w')

    glint_feature_dir = '/workspace/data/danchen'
    glint_feature_list = './feature_list/feature_list-27.txt'

    glint_fn = open(osp.join(glint_feature_dir, glint_feature_list)).readlines()
    for line in glint_fn:
        maxsim = -1
        last_label = ''
        line = line.strip()
        feat = load_feat(osp.join(glint_feature_dir,'glint_test_features_1862120/',  line))

        for jj  in range(len(lfw_mean)):
            mean = lfw_mean[jj]
            temp_sim = calc_similarity_cosine(feat, mean)
            print temp_sim
            if temp_sim > maxsim:
                maxsim = temp_sim
                last_label = sonfile_lfw[jj]
        if maxsim > 0.3:
            usefull_fn.writelines(line + '  ' + last_label + '  ' + str(maxsim))
            usefull_fn.writelines('\n')
