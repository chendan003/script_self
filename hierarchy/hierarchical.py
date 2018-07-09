import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
from data_reader import multiprocess_feature_data_reader
import shutil
import os
import os.path as osp
import sys
import argparse

def load_points(featureList):
    data, filePathList = multiprocess_feature_data_reader(featureList, nProcess=1)

# 
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataNameList', type=str, help='dataset name list')
    parser.add_argument('--featlist-dir', type=str, help='path to features')
    return parser.parse_args(argv)


def main(dataNameList, featurelist_dir):

 #   dataNameList = args.dataNameList

#    featurelist_dir = args.featlist_dir
    picSrcDir= ''
    
    for dataName in dataNameList:
	featlist = osp.join(featurelist_dir, '{0}_featlist.txt'.format(dataName))
	print 'featlist is', featlist
        data, filePathList = multiprocess_feature_data_reader(featlist)
    
        saveName = dataName
    # clustering
        for thresh in np.arange(0.5, 0.9, 0.02):
            print thresh
            clusters = hcluster.fclusterdata(data, thresh, metric="cosine", method='average', criterion="distance")
            print 'The number of clustered label is:',np.amax(clusters)
            print clusters
	    label_result = {}
	    for label in set(clusters):
	        label_num = np.sum(clusters==label)
	        label_result[label] = label_num

            for i in range(len(filePathList)):
                #picName = filePathList[i].split('/')[-1].replace('.npy', '')
                picName = filePathList[i].replace('.npy', '')
                srcPicPath = picSrcDir + picName
                lost_clustered_label = 'result_{0}/result_{1}/0'.format(saveName, thresh)

	        if label_result[clusters[i]] !=1:
                    try:
		        save_path = 'result_{0}/result_{1}/{2}'.format(saveName, thresh, clusters[i])
                        os.makedirs(save_path)
            	    except:
                        pass
                    shutil.copy(srcPicPath, save_path)

	        else:
		    if not osp.exists(lost_clustered_label):
		        os.makedirs(lost_clustered_label)
		    shutil.copy(srcPicPath, lost_clustered_label)
		   	


if __name__=='__main__':
    dataNameList = ['fangtan', 'life', 'liaozhai', 'museum', 'suzhouxinwen', 'shehuichuanzhen']    
    featurelist_dir ='/workspace/data/blued_code/ataraxia/inference/face/face-cluster/python/suzhoutai'
    main(dataNameList, featurelist_dir)
