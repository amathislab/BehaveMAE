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
from functools import partial, reduce
from operator import floordiv, mul
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general_hiera import GeneralizedHiera, HieraBlock
from .hiera_utils import conv_nd, undo_windowing


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


class HBehaveMAE(GeneralizedHiera):
    """Behavior Masked Autoencoder with Generalized Hiera backbone"""

    def __init__(
        self,
        in_chans: int = 1,
        patch_stride: Tuple[int, ...] = (2, 1, 3),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 1,
        decoder_num_heads: int = 1,
        decoding_strategy: str = "multi",
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        norm_loss: bool = False,
        **kwdargs,
    ):
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwdargs,
        )

        del self.norm, self.head
        encoder_dim_out = self.projections[-1].out_features
        self.encoder_norm = norm_layer(encoder_dim_out)

        overall_q_strides = list(
            map(lambda elements: reduce(mul, elements), zip(*self.q_strides))
        )
        self.mask_unit_spatial_shape_final = [
            i // s for i, s in zip(self.mask_unit_size, overall_q_strides)
        ]
        self.tokens_spatial_shape_final = [
            i // s for i, s in zip(self.tokens_spatial_shape, overall_q_strides)
        ]

        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for ix, i in enumerate(
            self.stage_ends[: self.q_pool]
        ):  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            # curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_strides[ix])]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_strides[0]))(
                    self.projections[ix].out_features,
                    self.projections[-1].out_features,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # patch stride of prediction
        # reconstruct every t-th frame, with t being the temporal stride of the initial patches
        self.pred_stride = (
            patch_stride[-1] * patch_stride[-2] * math.prod(overall_q_strides)
        )

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.pred_stride * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.decoding_strategy = decoding_strategy

        self.norm_loss = norm_loss

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patch_pixel_label_3d(self, input_vid, t, h, w):
        """
        Partitions a tensor of shape (B,C,T,H,W) into patches of shape (t, h, w).
        Parameters:
            input_tensor (torch.Tensor): The input tensor of shape (B, C, T, H, W).
        Returns:
            torch.Tensor: The output tensor reshaped to (B, C, T//t, H//h, W//w, t, h, w).
        """
        B, C, T, H, W = input_vid.shape

        # Ensure that the dimensions are divisible by the patch sizes
        assert (
            T % t == 0
        ), "T (num_timesteps) must be divisible by t (temporal patch size)"
        assert H % h == 0, "H (height) must be divisible by h (height patch size)"
        assert W % w == 0, "W (width) must be divisible by w (width patch size)"

        # Reshape the tensor to introduce the patch dimensions.
        # Here, we split each dimension T, H, W into (T//t, t), (H//h, h), (W//w, w) respectively.
        # This gives us a new shape of (B, C, T//t, t, H//h, h, W//w, w).
        output_tensor = input_vid.view(B, C, T // t, t, H // h, h, W // w, w)

        # Permute the dimensions to rearrange them into the desired shape.
        # We need to move the patch dimensions (t, h, w) to the end.
        # The target shape is (B, C, T//t, H//h, W//w, t, h, w).
        output_tensor = output_tensor.permute(0, 1, 2, 4, 6, 3, 5, 7)

        return output_tensor

    def get_label_3d(self, input_vid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, :: self.patch_stride[0], :, :]

        _, _, T, H, W = input_vid.shape
        t_num_blocks, h_num_blocks, w_num_blocks = self.tokens_spatial_shape_final

        label = self.patch_pixel_label_3d(
            input_vid, T // t_num_blocks, H // h_num_blocks, W // w_num_blocks
        )
        label = label.reshape(mask.shape[0], mask.shape[1], -1)

        label = label[mask]

        if self.norm_loss:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, return_intermediates=True)
        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # hacky: for multi-gpu training, set 'find_unused_parameters=True' in DDP
        if self.decoding_strategy == "single":
            # Use only the last layer's output for decoding
            x = intermediates[-1]

        else:
            # Multi-scale fusion
            x = 0.0
            for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
                x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        return x, mask

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_strides[0]) == 3:
            label = self.get_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]

        # MSE loss
        loss = (pred - label) ** 2

        return loss.mean(), pred, label, None

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.6,
        mask_strategy: str = "random",
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # add channel dimension
        x = x.unsqueeze(1)

        latent, mask = self.forward_encoder(x, mask_ratio, mask=mask)

        pred, pred_mask = self.forward_decoder(
            latent, mask
        )  # pred_mask is mask at resolution of *prediction*

        # Toggle mask, to generate labels for *masked* tokens
        return *self.forward_loss(x, pred, ~pred_mask), mask
