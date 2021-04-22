root_dir=/data/ls_png/
cur_dir=test_video_yt
c=0
for file in `find $root_dir/$cur_dir  | grep ".mp4"`
do
  file_list[$c]=$file
#   echo $file
  ((c++))
done

for file_path in ${file_list[*]}
do 
    # echo $file_path
    substr=${file_path##*/}
    step=${substr%*.mp4}
    echo ${substr} ${step}
    new_dir="$root_dir/$cur_dir/$step"
    echo $new_dir
    if [ ! -d $new_dir ]; then
        mkdir $new_dir
    fi

    ffmpeg -i $file_path  $new_dir/%6d.png
done