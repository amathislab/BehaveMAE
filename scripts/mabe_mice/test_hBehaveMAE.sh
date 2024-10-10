
experiment=experiment1

python run_test.py \
    --path_to_data_dir data/MABe22 \
    --dataset mabe_mice \
    --embedsum True \
    --fast_inference False \
    --batch_size 768 \
    --model gen_hiera \
    --input_size 900 3 24 \
    --stages 3 4 5 \
    --q_strides "5,1,1;1,3,1" \
    --mask_unit_attn True False False \
    --patch_kernel 3 1 24 \
    --init_embed_dim 128 \
    --init_num_heads 2 \
    --out_embed_dims 128 192 256 \
    --distributed \
    --num_frames 900 \
    --pin_mem \
    --num_workers 8 \
    --fill_holes False \
    --output_dir outputs/mice/${experiment}


cd hierAS-eval
            
nr_submissions=$(ls ../outputs/mice/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))

parallel --line-buffer \
    python evaluator.py \
        --task mabe_mice --output-dir results \
        --labels ../data/MABe22/mouse_triplets_test_labels.npy \
        --submission ../outputs/mice/${experiment}/test_submission_{}.npy \
    ::: "${files[@]}"
