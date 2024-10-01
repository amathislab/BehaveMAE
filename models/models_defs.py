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

from .general_hiera import GeneralizedHiera
from .hbehave_mae import HBehaveMAE


def gen_hiera(**kwdargs):
    return GeneralizedHiera(
        in_chans=1,
        embed_dim=kwdargs["init_embed_dim"],
        num_heads=kwdargs["init_num_heads"],
        patch_stride=kwdargs["patch_kernel"],
        patch_padding=(0, 0, 0),
        **kwdargs
    )


def hbehavemae(**kwdargs):
    return HBehaveMAE(
        in_chans=1,
        embed_dim=kwdargs["init_embed_dim"],
        num_heads=kwdargs["init_num_heads"],
        patch_stride=kwdargs["patch_kernel"],
        patch_padding=(0, 0, 0),
        **kwdargs
    )
