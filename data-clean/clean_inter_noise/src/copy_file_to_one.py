# -*- coding:utf-8 -*-
import os
import os.path as osp
import shutil
import sys

def copy_file(sourcefile, dstdir):
    if not osp.exists(sourcefile):
	print 'sourcefile doesn\'t exists'
    else:
	shutil.copy(sourcefile, dstdir)

def get_imglist(path):
    if osp.exists(path):
	imglist=os.listdir(path)
    else:
	imglist=[]
	print 'get_imglist occured wrong!'
    return imglist


if __name__=='__main__':
	merge-label-rsult-fn = sys.argv[1]
    img_dir = sys.argv[2]

    ff = open().readlines(merge-label-rsult-fn)
    for lines in ff:
	line= lines.strip().split()
	length = len(line)
	dstdir = osp.join(img_dir,line[0])
	for nn in range(1,length):
	    full_path = osp.join(img_dir, line[nn])
	    print full_path
	    imglist = get_imglist(full_path)
	    full_imglist = [osp.join(full_path, ii) for ii in imglist]
	    for tt in full_imglist:
		copy_file(tt, dstdir)
	    shutil.rmtree(full_path)
    
