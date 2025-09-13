python generate_super_resolution.py \
--model_path ./results/udf_super/epoch=499.ckpt \
--generate_method generate_meshes \
--npy_path ./outputs/udf_epoch=999.ckpt_batch_True \
--save_npy True \
--save_mesh False \
--level 0.0 \
--steps 20
