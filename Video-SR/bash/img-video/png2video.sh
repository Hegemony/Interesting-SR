input_dir=/ai_lab/daiqiuju/SR_VIDEO_DATASETS/NTIRE2020_quality_mapping_Vid3oc/TrainingSourceDomain
output_dir=/ai_lab/daiqiuju/SR_VIDEO_DATASETS/NTIRE2020_quality_mapping_Vid3oc/video_cmp
## บ๓ืบ
sub_pix=sc

if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

for temp_name in {000..059}
do 
    echo $temp_name
    imgs_path=${input_dir}/${temp_name}/%06d.png   
    
    ffmpeg -i ${input_dir}/${temp_name}/%06d.png  ${output_dir}/${temp_name}${sub_pix}.mp4    
done