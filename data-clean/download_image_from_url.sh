# !/bin/bash
# how to use the script, example: sh split_imglist.sh 4 , the parameter means lines number of every file

file=img.txt
evefile_lines=$1

total_lines=`sed -n '$=' ${file}`
echo 'Total lines number is: '$total_lines

n=$[total_lines/evefile_lines+1]
echo 'Create files number is:' $n

start=1
echo $start
for i in `seq 1 $n`;
do
    end=$[start+evefile_lines-1]
    echo $end
    txtfile=image-$i.txt
    sed -n  $start,${end}\p $file  > $txtfile
    save_img_dir=save_image_$i
    wget -i $txtfile  -P $save_img_dir
    start=$[$end+1]
done
~
