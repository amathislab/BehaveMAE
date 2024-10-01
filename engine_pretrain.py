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
import math
from typing import Iterable

import torch
from iopath.common.file_io import g_pathmgr as pathmgr

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    data_loader_val: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    fp32=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(
    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples, targets = samples

        samples = samples.to(device, non_blocking=True)
        if targets:
            targets = [tgt.to(device, non_blocking=True) for tgt in targets]
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        mask_ratio = args.mask_ratio

        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, _, _, _, _ = model(
                samples,
                targets,
                mask_ratio=mask_ratio,
                mask_strategy=args.masking_strategy,
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as _:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=args.mask_ratio)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    if data_loader_val:
        # compute loss on test data
        if epoch % 5 == 0:
            for data_iter_step, samples in enumerate(
                metric_logger.log_every(data_loader_val, print_freq, header)
            ):
                samples, targets = samples
                samples = samples.to(device, non_blocking=True)
                if targets:
                    targets = [tgt.to(device, non_blocking=True) for tgt in targets]
                if len(samples.shape) == 6:
                    b, r, c, t, h, w = samples.shape
                    samples = samples.reshape(b * r, c, t, h, w)
                mask_ratio = args.mask_ratio

                with torch.no_grad():
                    loss, _, _, _, _ = model(
                        samples,
                        targets,
                        mask_ratio=mask_ratio,
                        mask_strategy=args.masking_strategy,
                    )

                loss_value = loss.item()
                loss /= accum_iter
                torch.cuda.synchronize()
                loss_value_reduce = misc.all_reduce_mean(loss_value)
                if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                    """We use epoch_1000x as the x-axis in tensorboard.
                    This calibrates different curves when batch size changes.
                    """
                    epoch_1000x = int(
                        (data_iter_step / len(data_loader) + epoch) * 1000
                    )
                    log_writer.add_scalar("test_loss", loss_value_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
