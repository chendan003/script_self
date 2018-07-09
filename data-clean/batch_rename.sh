#!/bin/bash
for files in `ls ./image`
do 
    dir=/Users/dannychen/test_file/image/ 
    second=${dir}${files}

    for img in `ls ${second}`
    do
        bname=${files}
        imgname=${img}
    # 拼接成文件名
        filename=$bname$imgname
    # 更改文件名
        org=${second}/${img}
        new=${second}/${filename}
        mv  $org $new
    done
done



