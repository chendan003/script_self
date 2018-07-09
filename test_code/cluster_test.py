#coding=utf-8
from __future__ import division
from my_cluster import *
import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_label(testsetDir):
    alone_cluster_label_cnt = 1000
    labelDict = {}
    for dir in os.listdir(testsetDir):
        if dir.startswith('.'):
            continue
        else:
            currentDir = os.path.join(testsetDir, dir)
            for fileName in os.listdir(currentDir):
                if fileName.endswith('.jpg'):
                    if str(dir) != '0':
                        labelDict[fileName] = dir
                    else:
                        labelDict[fileName] = str(alone_cluster_label_cnt)
                        alone_cluster_label_cnt += 1
    return labelDict


def test_from_result(resultDict, labelDict, methodList=['DBSCAN'], picDir=None):
    resultClusterDict = make_clusterDict_from_resultDict(resultDict)
    labelClusterDict = make_clusterDict_from_resultDict(labelDict)
    f_score, precision, recall = pairwise_f_score(resultClusterDict, labelClusterDict, labelDict)
    return f_score, precision, recall

def cluster_and_test_from_video_dir(featuresDir, picDir, labelDict, methodList=['DBSCAN']):
    if methodList[0] == 'API':
        methodResultDict = {}
        methodResultDict['API'] = test_former_api(featuresDir)
    else:    
        methodResultDict = cluster_from_video_dir(featuresDir, picDir, methodList)
    for method in methodResultDict.keys():
        resultDict = methodResultDict[method]
        f_score, precision, recall = test_from_result(resultDict, labelDict)
        return f_score, precision, recall

def find_max_clustered_num(label, resultClusterDict, labelDict):
    maxNum = 0
    for k,cluster in resultClusterDict.items():
        num = count_label_in_cluster(label, cluster, labelDict)
        if num > maxNum:
            maxNum = num
    return maxNum

def count_label_in_cluster(label, cluster, labelDict):
    num = 0
    for x in cluster:
        if labelDict[x] == label:
            num += 1
    return num

def make_clusterDict_from_resultDict(resultDict):
    clusterDict = {}
    alone_cluster_label_cnt = 1000
    for key in resultDict.keys():
        targetKey = resultDict[key]
        if targetKey == '-1':
            targetKey = str(alone_cluster_label_cnt)
            alone_cluster_label_cnt += 1
        if not clusterDict.has_key(targetKey):
            clusterDict[targetKey] = []
        clusterDict[targetKey].append(key)
    return clusterDict

def find_cluster_most_label(listOfSameCluster, labelDict):
    _statLabelDict = {}
    for fileName in listOfSameCluster:
        if not _statLabelDict.has_key(labelDict[fileName]):
            _statLabelDict[labelDict[fileName]] = 1
        else:
            _statLabelDict[labelDict[fileName]] += 1
    mostLabel = -1
    mostLabelNum = -1
    for k,v in _statLabelDict.items():
        if v > mostLabelNum:
            mostLabelNum = v
            mostLabel = k
    return mostLabel, mostLabelNum

    
def test_former_api(featuresDir):
    classCnt = 0
    resultDict = {}
    for dirName in os.listdir(featuresDir):
        if str(dirName).startswith('.'):
            continue
        currentDir = os.path.join(featuresDir, dirName)
        for fileName in os.listdir(currentDir):
            if fileName.endswith('.jpg'):
                resultDict[fileName] = str(classCnt)
        classCnt += 1
    return resultDict

def pairwise_f_score(resultClusterDict, labelClusterDict, labelDict):
    precision = pairwise_precision(resultClusterDict, labelDict) 
    recall = pairwise_recall(resultClusterDict, labelClusterDict, labelDict)
    if precision == 0 and recall == 0:
        return 0, precision, recall
    f_score = 2 * precision * recall / (precision + recall)
    # print precision, recall
    return f_score, precision, recall

def pairwise_precision(resultClusterDict, labelDict):
    above = 0
    below = 0
    for k, cluster in resultClusterDict.items():
        below += compute_combination(2, len(cluster))

        above += count_right_pair_in_result_cluster(cluster, labelDict)
    if above == 0 and below == 0:
        return 0
    return above / below

def pairwise_recall(resultClusterDict, labelClusterDict, labelDict):
    above = 0
    below = 0
    for k, cluster in labelClusterDict.items():
        below += compute_combination(2, len(cluster))
    for k, cluster in resultClusterDict.items():
        above += count_right_pair_in_result_cluster(cluster, labelDict)
    return above / below

def count_right_pair_in_result_cluster(cluster, labelDict):
    num = 0
    cntDict = {}
    for x in cluster:
        if cntDict.has_key(labelDict[x]):
            cntDict[labelDict[x]] += 1
        else:
            cntDict[labelDict[x]] = 1
    
    for k, v in cntDict.items():
        num += compute_combination(2, v)
    return num

def compute_combination(upNum, downNum):
    above = 1
    below = 1
    for i in range(upNum):
        above *= downNum - i
        below *= upNum - i
    return above / below




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clustering and Pairwise F_score Evaluation')
    parser.add_argument('--method', type=str, required=False, help='DBSCAN, API, AP, RankOrder')
    parser.add_argument('--labelDir', type=str, required=False, default='/workspace/data/blued_code/test_cluster_result/life_standard', help='Path of labeled pictures')
    parser.add_argument('--featuresDir', type=str, required=False, help='Path of features to be clustered')
    parser.add_argument('--resultDir', type=str, required=False, default='/workspace/data/blued_code/test_cluster_result/life_0.5', help='Test from result')
    parser.add_argument('--featureList', type=str, required=False, help='Feature list of feature file name')
    parser.add_argument('--picDir', type=str, required=False, help='Path of pictures to be clustered')
    parser.add_argument('--saveResult', type=bool, required=False, help='Whether to save the result pics')
    parser.add_argument('--saveDir', type=str, required=False, help='Path to save clustered pictures')
    parser.add_argument('--videoName', type=str, required=False, default='Hierarchy_suzhouxinwen', help='Name of the clustered video')
    parser.add_argument('--eps', type=float, required=False, default=None, help='DBSCAN parameter')
    parser.add_argument('--nProcess', type=int, required=False, default=1, help='Number of processes to read data')
    args = vars(parser.parse_args())


    name = args['videoName']
    _resultSuffixList = np.arange(0.2, 0.9, 0.02)
    for i in range(len(_resultSuffixList)):
        _resultSuffixList[i] = round(_resultSuffixList[i], 2)


    f_scoreList = []
    precisionList = []
    recallList = []
    for _resultSuffix in _resultSuffixList:
        labelDict = load_label(args['labelDir'])
        resultDict = load_label(args['resultDir'])#+str(_resultSuffix))
        f_score, precision, recall = test_from_result(resultDict, labelDict)
        f_scoreList.append(f_score)
        precisionList.append(precision)
        recallList.append(recall)
        print "suffix:", _resultSuffix, f_score, precision, recall

    d = {'video_name': [name]*35, 'eps': _resultSuffixList, 'F_score': f_scoreList, 'precision': precisionList, 'recall': recallList}
    df = pd.DataFrame(data=d)
    cols=['video_name','eps','F_score','precision','recall']
    df=df.ix[:,cols]
    df.to_csv('{0}video_test_result_SUZHOU.csv'.format(name), encoding = "utf-8")
    print df
    
    plt.xlabel('eps')  
    plt.ylabel('Score')  
    
    plt.plot(_resultSuffixList, f_scoreList,'g', label='Trainset Accuracy')  
    plt.plot(_resultSuffixList, precisionList,'b', label='Testset Accuracy')
    plt.plot(_resultSuffixList, recallList,'r', label='Testset Accuracy')
    plt.grid()  
    plt.savefig('{0}_test_plot_SUZHOU'.format(name))
