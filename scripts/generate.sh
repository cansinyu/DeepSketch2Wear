#! /bin/bash
python generate_udf.py \
--generate_method generate_based_on_sketch_batch \
--model_path results/udf_power/epoch=999.ckpt \
--sketch_file /public/home/chenxinyu2023/3D/data/step2_data/test.txt \
--output_path outputs \
--num_generate 3 \
--steps 20 

