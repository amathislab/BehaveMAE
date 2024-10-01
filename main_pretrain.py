# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Original Work:
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya et al.
# https://arxiv.org/abs/2306.00989/
#
# Enhanced and modified by Stoffl et al.
#
# For more details on our work, please refer to:
# Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders
# Lucas Stoffl, Andy Bonnetto, Stéphane d'Ascoli, Alexander Mathis
# https://www.biorxiv.org/content/10.1101/2024.08.06.606796v1
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from torch.utils.tensorboard import SummaryWriter

from datasets.hbabel import hBABELDataset
from datasets.mabe22_mice import MABeMouseDataset
from datasets.shot7m2 import SHOT7M2Dataset
from engine_pretrain import train_one_epoch
from models import models_defs
from util import misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import parse_tuples, str2bool


def get_args_parser():
    parser = argparse.ArgumentParser("hBehaveMAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    parser.add_argument(
        "--dataset",
        default="shot7m2",
        type=str,
        help="Type of dataset [shot7m2, mabe_mice, hbabel]",
    )

    parser.add_argument("--sliding_window", default=1, type=int)
    parser.add_argument("--fill_holes", default=False, type=str2bool)
    parser.add_argument("--data_augment", default=False, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")
    parser.add_argument("--include_test_data", action="store_true")

    # for AMASS / hBABEL loading
    parser.add_argument("--joints3d_procrustes", default=True, type=str2bool)

    # Model parameters
    parser.add_argument(
        "--model",
        default="hbehavemae",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    # train a non-hierarchical model ("BehaveMAE")
    parser.add_argument("--non_hierarchical", default=False, type=str2bool)

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )
    parser.add_argument(
        "--masking_strategy",
        default="random",
        type=str,
    )
    parser.add_argument(
        "--decoding_strategy",
        default="multi",
        type=str,
        help="Decoding strategy for combining latents [multi, single]",
    )
    parser.add_argument("--decoder_embed_dim", default=128, type=int)
    parser.add_argument("--decoder_depth", default=1, type=int)
    parser.add_argument("--decoder_num_heads", default=1, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=20, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--distributed", action="store_true")

    # hBehaveMAE specific parameters
    parser.add_argument("--input_size", default=(600, 3, 24), nargs="+", type=int)
    parser.add_argument("--stages", default=(2, 3, 4), nargs="+", type=int)
    parser.add_argument(
        "--q_strides", default=[(1, 1, 3), (1, 1, 4), (1, 3, 1)], type=parse_tuples
    )
    parser.add_argument(
        "--mask_unit_attn", default=[True, False, False], nargs="+", type=str2bool
    )
    parser.add_argument("--patch_kernel", default=(4, 1, 2), nargs="+", type=int)
    parser.add_argument("--init_embed_dim", default=48, type=int)
    parser.add_argument("--init_num_heads", default=2, type=int)
    parser.add_argument("--out_embed_dims", default=(32, 64, 96), nargs="+", type=int)

    parser.add_argument("--norm_loss", default=True, type=str2bool)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--path_to_data_dir",
        default="",
        help="path where to load data from",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save",
    )
    parser.add_argument(
        "--log_dir",
        default="./log_dir",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")

    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
    )
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--trunc_init",
        action="store_true",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.set_defaults(fp32=True)

    parser.add_argument(
        "--beta",
        default=None,
        type=float,
        nargs="+",
    )
    return parser


def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.dataset.lower().startswith("shot"):
        dataset_train = SHOT7M2Dataset(
            mode="pretrain",
            path_to_data_dir=args.path_to_data_dir,
            num_frames=args.num_frames,
            sliding_window=args.sliding_window,
            sampling_rate=args.sampling_rate,
            augmentations=args.data_augment,
            include_testdata=args.include_test_data,
            split_tokenization=True,
            patch_size=args.patch_kernel,
            q_strides=args.q_strides,
        )
        dataset_test = SHOT7M2Dataset(
            mode="test",
            path_to_data_dir=args.path_to_data_dir,
            num_frames=args.num_frames,
            sliding_window=args.sliding_window,
            sampling_rate=args.sampling_rate,
            augmentations=None,
            split_tokenization=True,
        )
    elif args.dataset == "hbabel":
        dataset_train = hBABELDataset(
            mode="pretrain",
            path_to_data_dir=args.path_to_data_dir,
            joints3d_procrustes=args.joints3d_procrustes,
            num_frames=args.num_frames,
            sliding_window=args.sliding_window,
            sampling_rate=args.sampling_rate,
            augmentations=args.data_augment,
            include_testdata=args.include_test_data,
            patch_size=args.patch_kernel,
            q_strides=args.q_strides,
        )
        dataset_test = hBABELDataset(
            mode="test",
            path_to_data_dir=args.path_to_data_dir,
            joints3d_procrustes=args.joints3d_procrustes,
            num_frames=args.num_frames,
            sliding_window=args.sliding_window,
            sampling_rate=args.sampling_rate,
        )
    elif args.dataset == "mabe_mice":
        dataset_train = MABeMouseDataset(
            mode="pretrain",
            path_to_data_dir=args.path_to_data_dir,
            num_frames=args.num_frames,
            sliding_window=args.sliding_window,
            sampling_rate=args.sampling_rate,
            fill_holes=args.fill_holes,
            augmentations=args.data_augment,
            centeralign=args.centeralign,
            include_testdata=args.include_test_data,
            patch_size=args.patch_kernel,
            q_strides=args.q_strides,
        )
        dataset_test = MABeMouseDataset(
            mode="test",
            path_to_data_dir=args.path_to_data_dir,
            num_frames=args.num_frames,
            sliding_window=args.sliding_window,
            sampling_rate=args.sampling_rate,
            fill_holes=args.fill_holes,
            augmentations=None,
            centeralign=args.centeralign,
        )
    else:
        print(f"Dataset {args.dataset} unknown...")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if dataset_test:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        if dataset_test:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        try:
            pathmgr.mkdirs(args.log_dir)
        except Exception as _:
            pass
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if dataset_test:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_test = None

    # define the model
    model = models_defs.__dict__[args.model](
        **vars(args),
    )
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        fup = True if args.decoding_strategy == "single" else False
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=fup,
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )
    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = args.beta
    optimizer = torch.optim._multi_tensor.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            data_loader_test,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32,
        )
        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def launch_one_thread(
    local_rank,
    shard_rank,
    num_gpus_per_node,
    num_shards,
    init_method,
    output_path,
    opts,
    stats_queue,
):
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)
