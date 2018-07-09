#!/bin/bash  
  
for a in `find ./ -type f -name "*.ts"`  
do  
  
cd $(dirname ${a})  
dir=`pwd |awk -F "/" '{print $NF}'`  
  
if [ ! -d "$dir" ]; then  
    mkdir -p  $1/images/${dir}/$(basename ${a})
fi  
    echo $(date) >> ffmpeg.log  
    echo -e "start convert $a " >> ffmpeg.log  
ffmpeg -i $a -r $2  $1/images/${dir}/$(basename ${a})/image_%05d.jpg >> ffmpeg.log  
##basename  获取文件本身名称，对应的  dirname  则为获取该文件的路径  
  
if [ $? -eq 0  ];then   
    echo -e "convert $(basename ${a}) done" >> ffmpeg.log  
else   
    echo -e "convert $(basename ${a}) failed" >> ffmpeg.log  
fi  
    # echo -e "done \n"   >> ffmpeg.log  
      
done  
