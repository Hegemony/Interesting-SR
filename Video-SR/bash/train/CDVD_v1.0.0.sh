python  ../../main.py --experiment_dir /ai_lab/daiqiuju/outdir_deblur/CDVD_V1.0.0 \
--dir_data_source /ai_lab/daiqiuju/SR_VIDEO_DATASETS/NTIRE2020_quality_mapping_Vid3oc/TrainingSourceDomain \
--dir_data_gt /ai_lab/daiqiuju/SR_VIDEO_DATASETS/NTIRE2020_quality_mapping_Vid3oc/TrainingTargetDomain \
--dir_data_test_source /ai_lab/daiqiuju/SR_VIDEO_DATASETS/NTIRE2020_quality_mapping_Vid3oc/sub_valid_source \
--dir_data_test_gt /ai_lab/daiqiuju/SR_VIDEO_DATASETS/NTIRE2020_quality_mapping_Vid3oc/sub_valid_target \
--pre_train_flow_net "" \
--pre_train_CDVD "" \
--epochs 100 --batch_size 8 \
--n_GPUs 2 \
--data_train DVD \
--data_test DVD
