
# number of gpus to use
GPUS=$1

common_args="--dataset mabe_mice \
    --path_to_data_dir data/MABe22/mouse_triplet_train.npy \
    --batch_size 768 \
    --model hbehavemae \
    --input_size 900 3 24 \
    --stages 3 4 5 \
    --q_strides 5,1,1;1,3,1 \
    --mask_unit_attn True False False \
    --patch_kernel 3 1 24 \
    --init_embed_dim 128 \
    --init_num_heads 2 \
    --out_embed_dims 128 192 256 \
    --epochs 200 \
    --num_frames 900 \
    --decoding_strategy multi \
    --decoder_embed_dim 128 \
    --decoder_depth 1 \
    --decoder_num_heads 1 \
    --pin_mem \
    --num_workers 8 \
    --sliding_window 17 \
    --blr 1.6e-4 \
    --warmup_epochs 40 \
    --masking_strategy random \
    --mask_ratio 0.875 \
    --clip_grad 0.02 \
    --checkpoint_period 20 \
    --fill_holes False \
    --data_augment True \
    --norm_loss True \
    --seed 0 \
    --output_dir outputs/mice/experiment1 \
    --log_dir logs/mice/experiment1"


if [[ $GPUS == 1 ]]; then
    OMP_NUM_THREADS=1 python run_pretrain.py $common_args
else
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --node_rank 0 --master_addr=127.0.0.1 --master_port=2999 \
        run_pretrain.py --distributed $common_args
fi
