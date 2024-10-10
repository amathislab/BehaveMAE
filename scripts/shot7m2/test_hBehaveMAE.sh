
experiment=experiment1

python run_test.py \
    --path_to_data_dir data/Shot7M2/test/test_dictionary_poses.npy \
    --dataset shot7m2 \
    --embedsum False \
    --fast_inference False \
    --batch_size 512 \
    --model gen_hiera \
    --input_size 400 1 72 \
    --stages 2 3 4 \
    --q_strides "2,1,4;2,1,6" \
    --mask_unit_attn True False False \
    --patch_kernel 2 1 3 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 78 128 256 \
    --distributed \
    --num_frames 400 \
    --pin_mem \
    --num_workers 8 \
    --output_dir outputs/shot7m2/${experiment}


cd hierAS-eval

nr_submissions=$(ls ../outputs/shot7m2/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))

parallel --line-buffer \
    python evaluator.py \
        --task Shot7M2 --output-dir results \
        --labels ../data/Shot7M2/test/benchmark_labels.npy \
        --submission ../outputs/shot7m2/${experiment}/test_submission_{}.npy \
    ::: "${files[@]}"
