
experiment=experiment1

python run_test.py \
    --path_to_data_dir data/babel \
    --dataset hbabel \
    --joints3d_procrustes True \
    --embedsum False \
    --fast_inference False \
    --batch_size 128 \
    --model gen_hiera \
    --input_size 450 1 75 \
    --stages 2 3 4 \
    --q_strides "5,1,1;3,1,1" \
    --mask_unit_attn True False False \
    --patch_kernel 3 1 75 \
    --init_embed_dim 128 \
    --init_num_heads 2 \
    --out_embed_dims 64 96 128 \
    --distributed \
    --num_frames 450 \
    --pin_mem \
    --num_workers 8 \
    --output_dir outputs/hBABEL/${experiment}
