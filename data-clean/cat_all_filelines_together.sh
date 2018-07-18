# !/usr/bash

for file in `ls $1`;
do
    full_path='/workspace/data/danchen/'$1$file
    echo $full_path
    cat $full_path >> $2
done
