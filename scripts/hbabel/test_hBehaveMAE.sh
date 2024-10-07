
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


cd hierAS-eval

nr_submissions=$(ls ../outputs/hBABEL/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))

parallel --line-buffer \
    python evaluator.py \
        --task hBABEL --output-dir results \
        --labels ../data/hBABEL/hbabel_val_test_actions_val_top_120_60_filtered.npy \
        --submission ../outputs/hBABEL/${experiment}/test_submission_{}.npy \
    ::: "${files[@]}"
