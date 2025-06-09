#! /bin/bash
python train_super_resolution.py \
--name udf_super \
--batch_size 64 \
--new False \
--continue_training Ture \
--training_epoch 500  \
--split_dataset False \
--save_every_epoch 50 \
--udf_folder /home/ubuntu/public_c/cxy/data/las_udf/udf
