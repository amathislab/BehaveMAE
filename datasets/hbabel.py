# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# For more details on our work, please refer to:
# Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders
# Lucas Stoffl, Andy Bonnetto, Stéphane d'Ascoli, Alexander Mathis
# https://www.biorxiv.org/content/10.1101/2024.08.06.606796v1
# --------------------------------------------------------

import __future__

import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from torchvision import transforms

from .augmentations import RandomMove, RandomShift
from .pose_traj_dataset import BasePoseTrajDataset


class hBABELDataset(BasePoseTrajDataset):
    """
    hBABEL dataset: Bodies, Action and Behavior with English Labels
    """

    DEFAULT_FRAME_RATE = 30
    NUM_KEYPOINTS = 25
    KPTS_DIMENSIONS = 3
    NUM_INDIVIDUALS = 1
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)

    NTU_KPTS = np.array(
        [
            0, 3, 12, 15,
            16, 18, 20, 22,  # left hand
            17, 19, 21, 37,  # right hand
            1, 4, 7, 10,  # left leg
            2, 5, 8, 11,  # right hand
            9, 63, 64, 68, 69,
        ],
        dtype=np.int32,
    )

    NTU_KPT_GROUPING = np.array(
        [
            5, 6, 7, 21, 22,  # left arm/hand
            9, 10, 11, 23, 24,  # right arm/hand
            4, 12, 13, 14, 15,  # left leg + shoulder
            8, 16, 17, 18, 19,  # right leg + shoulder
            0, 1, 2, 3, 20,  # head, spine
        ],
        dtype=np.int32,
    )

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 100,
        sliding_window: int = 1,
        include_testdata: bool = False,
        augmentations: transforms.Compose = None,
        gender: str = "male",
        # pose_rep: str = "3djoints",
        joints3d_procrustes: bool = True,
        **kwargs,
    ):
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.num_frames = num_frames
        self.sample_frequency = self.DEFAULT_FRAME_RATE  # downsample frames if needed

        self.mode = mode
        self.gender = gender

        self.joints3d_procrustes = joints3d_procrustes

        if augmentations:
            self.augmentations = transforms.Compose(
                [
                    RandomShift(p=0.2),
                    RandomMove(p=0.4),
                ]
            )
        else:
            self.augmentations = None

        self.load_data(include_testdata)

        self.preprocess()

    def load_data(self, include_testdata) -> None:
        """Loads dataset"""
        self.path = Path(self.path)
        if self.mode == "pretrain":
            self.raw_data = joblib.load(
                self.path
                / f"babel-smplh-30fps-{self.gender}/train_proc_realigned_procrustes.pth.tar"
            )
            if include_testdata:
                raw_data_test = joblib.load(
                    self.path
                    / f"babel-smplh-30fps-{self.gender}/test_proc_realigned_procrustes.pth.tar"
                )
                self.raw_data.extend(raw_data_test)
        elif self.mode == "test":
            self.raw_data = joblib.load(
                self.path
                / f"babel-smplh-30fps-{self.gender}/val_proc_realigned_procrustes.pth.tar"
            )

    def featurise_keypoints(self, keypoints):
        keypoints = torch.from_numpy(keypoints).to(torch.float32)
        # group keypoints for fusing reasonable tokens
        keypoints = keypoints.reshape(
            self.max_keypoints_len, self.NUM_KEYPOINTS, self.KPTS_DIMENSIONS
        )
        keypoints = keypoints[:, self.NTU_KPT_GROUPING, :].reshape(
            self.max_keypoints_len, -1
        )
        return keypoints

    def preprocess(self):
        """
        Does initial preprocessing on entire dataset.
        """

        motion_data = []

        sample_ids = []
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window

        for idx, sample in enumerate(self.raw_data):

            seq_len = len(sample["poses"])

            if seq_len < sub_seq_length:
                pad_len = sub_seq_length - seq_len
                seq_len = sub_seq_length
            else:
                pad_len = 0

            if self.joints3d_procrustes:
                features = sample["joint_positions_processed"]

            else:  # (-> raw NTU kpts)
                features = sample["joint_positions"][:, self.NTU_KPTS, :]

                # Prep. data for normalization
                features = features.transpose(2, 0, 1)  # C, T, V
                features = features[:, :, :, np.newaxis]  # C, T, V, M

                # Normalize (pre-process) in NTU RGBD-style
                features = self.ntu_pre_normalization(features)
                features = features.transpose(1, 2, 3, 0).squeeze()  # T, V, C

            features = features.reshape(features.shape[0], -1)
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="edge")
            motion_data.append(features)

            sample_ids.extend(
                [
                    (idx, i)
                    for i in np.arange(0, seq_len - sub_seq_length + 1, sliding_window)
                ]
            )

        self.motion_data = motion_data

        self.items = list(np.arange(len(sample_ids)))
        self.sample_ids = sample_ids
        self.keypoints_ids = self.sample_ids
        self.n_frames = len(self.sample_ids)

        # self.nfeats = 135

        del self.raw_data

    @staticmethod
    def ntu_pre_normalization(
        data,
        top_ind=[1, 2, 3],
        bot_ind=[0, 12, 16],
        right_ind=[8, 16],
        left_ind=[4, 12],
    ):
        C, T, V, M = data.shape
        skeleton = np.transpose(data, [3, 1, 2, 0])  # C, T, V, M  to  M, T, V, C

        # print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
        if skeleton.sum() != 0:
            main_body_center = skeleton[0][:, 1:2, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask

        # print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
        if skeleton.sum() != 0:
            # joint_bottom = skeleton[0, 0, zaxis[0]]
            joint_bottom = np.mean(skeleton[0, 0, bot_ind], axis=0)
            joint_top = np.mean(skeleton[0, 0, top_ind], axis=0)
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = hBABELDataset.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = hBABELDataset.rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        skeleton[i_p, i_f, i_j] = np.dot(matrix_z, joint)

        # print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
        if skeleton.sum() != 0:
            joint_rshoulder = np.mean(skeleton[0, 0, right_ind], axis=0)
            joint_lshoulder = np.mean(skeleton[0, 0, left_ind], axis=0)
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = hBABELDataset.angle_between(
                joint_rshoulder - joint_lshoulder, [1, 0, 0]
            )
            matrix_x = hBABELDataset.rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        skeleton[i_p, i_f, i_j] = np.dot(matrix_x, joint)
        data = np.transpose(skeleton, [3, 1, 2, 0])

        return data

    @staticmethod
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )

    @staticmethod
    def unit_vector(vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angle_between(v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = hBABELDataset.unit_vector(v1)
        v2_u = hBABELDataset.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def fill_holes():
        pass

    def normalize(self, data):
        shift = torch.load("datasets/babel_stats/ntu_feats_mean.pt")
        scale = torch.load("datasets/babel_stats/ntu_feats_std.pt")
        return torch.nan_to_num((data - shift) / scale)

    @staticmethod
    def _normalize(data):
        shift = torch.load("datasets/babel_stats/ntu_feats_mean.pt")
        scale = torch.load("datasets/babel_stats/ntu_feats_std.pt")
        if not torch.is_tensor(data):
            shift = shift.numpy()
            scale = scale.numpy()
            return np.nan_to_num((data - shift) / scale)
        return torch.nan_to_num((data - shift) / scale)

    def unnormalize(self):
        pass

    def transform_to_centered_data(self):
        pass

    def transform_to_centeralign_components(self):
        pass

    def prepare_subsequence_sample(self, sequence: np.ndarray):
        """
        Prepares training sample
        """

        if self.augmentations:
            sequence = sequence.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            sequence = self.augmentations(sequence)
            sequence = sequence.reshape(self.max_keypoints_len, -1)
            # print('Warning: no augmentations implemented for hBABEL')

        feats = self.featurise_keypoints(sequence)

        feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)

        return feats

    def __getitem__(self, idx: int):

        subseq_ix = self.sample_ids[idx]
        subsequence = self.motion_data[subseq_ix[0]][
            subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len
        ]

        inputs = self.prepare_subsequence_sample(subsequence)

        return inputs, []
