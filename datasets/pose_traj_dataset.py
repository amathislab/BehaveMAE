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

import copy
from abc import abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.utils.data

# Some functions are adapted from TREBA
# "TREBA" by Sun, Jennifer J and Kennedy, Ann and Zhan, Eric and Anderson, David J and Yue, Yisong and Perona, Pietro is licensed under CC BY-NC-SA 4.0 license.
# https://github.com/neuroethology/TREBA/blob/c522e169738f5225298cd4577e5df9085130ce8a/util/datasets/mouse_v1/augmentations/augmentation_functions.py


class BasePoseTrajDataset(torch.utils.data.Dataset):
    """
    Primary Pose Trajectory (+Features) dataset.
    """

    DEFAULT_FRAME_RATE = None
    DEFAULT_GRID_SIZE = None
    NUM_INDIVIDUALS = None
    NUM_KEYPOINTS = None
    KPTS_DIMENSIONS = None
    DEFAULT_NUM_TRAINING_POINTS = None
    DEFAULT_NUM_TESTING_POINTS = None
    SAMPLE_LEN = None
    KEYFRAME_SHAPE = None
    NUM_TASKS = None
    STR_BODY_PARTS = None
    BODY_PART_2_INDEX = None

    def __init__(
        self,
        path_to_data_dir: Path,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        interp_holes: bool = False,
        patch_size: tuple = (4, 1, 2),
        q_strides: list = [(1, 1, 3), (1, 1, 4), (1, 3, 1)],
        **kwargs
    ):

        self.path = path_to_data_dir
        self.scale = scale

        # defined if data has been loaded
        self.has_annotations = None
        self.annotation_names = []
        self.annotations = {}

        # defined when data has been preprocessed.
        self.items = None
        self.keypoints = None
        self.n_frames = None

        self.max_keypoints_len = num_frames
        self.max_seq_length = num_frames
        self.sliding_window = sliding_window

        self.augmentations = None

        self._sampling_rate = sampling_rate

        self.interp_holes = interp_holes

        self.patch_size = patch_size

    def get_kwargs(self) -> dict:
        """returns positional arguments"""
        return {
            "path": self.path,
            "frame_rate": self.frame_rate,
            "sample_frequency": self.sample_frequency,
            "flatten": self.flatten,
            "scale": self.scale,
        }

    @abstractmethod
    def load_data(self, include_testdata, include_round2data) -> None:
        pass

    @abstractmethod
    def load_labeled_data(self) -> None:
        pass

    def check_annotations(self) -> None:
        """Annotation check handler"""
        self.has_annotations = "vocabulary" in self.raw_data.keys()
        if self.has_annotations:
            self.annotation_names = self.raw_data["vocabulary"]

    @abstractmethod
    def featurise_keypoints(self, keypoints):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    def sample_random_sequence(self):
        """Collect a training sequence at random"""
        idx = np.random.randint(0, len(self))
        return self.keypoints[idx]

    def sample_random_keypoints(self):
        """Collect a training sample at random"""
        seq = self.sample_random_sequence()
        seq_idx = np.random.randint(0, len(seq))
        feats = self.featurise_keypoints(seq[None, seq_idx])
        feats = feats.reshape(1, -1)
        return feats

    @staticmethod
    def downsample(keypoints: np.ndarray, sample_frequency) -> np.ndarray:
        """Downsamples frames"""
        return keypoints[:, ::sample_frequency, ...]

    def get_num_frames(self):
        return self.max_keypoints_len

    def __len__(self):
        return len(self.keypoints_ids)

    @staticmethod
    def fill_holes(data):
        clean_data = copy.deepcopy(data)
        num_individuals = clean_data.shape[1]
        for m in range(num_individuals):
            holes = np.where(clean_data[0, m, :, 0] == 0)
            if not holes:
                continue
            for h in holes[0]:
                sub = np.where(clean_data[:, m, h, 0] != 0)
                if sub and sub[0].size > 0:
                    clean_data[0, m, h, :] = clean_data[sub[0][0], m, h, :]
                # else:
                #     return np.empty((0))

        for fr in range(1, np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr, m, :, 0] == 0)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr, m, h, :] = clean_data[fr - 1, m, h, :]
        return clean_data

    def normalize(self, data):
        """Scale by dimensions of image and mean-shift to center of image."""
        state_dim = data.shape[1] // 2
        shift = [
            int(self.DEFAULT_GRID_SIZE / 2),
            int(self.DEFAULT_GRID_SIZE / 2),
        ] * state_dim
        scale = [
            int(self.DEFAULT_GRID_SIZE / 2),
            int(self.DEFAULT_GRID_SIZE / 2),
        ] * state_dim
        return np.divide(data - shift, scale)

    @staticmethod
    def _normalize(data, grid_size):
        """Scale by dimensions of image and mean-shift to center of image."""
        state_dim = data.shape[1] // 2
        shift = [int(grid_size / 2), int(grid_size / 2)] * state_dim
        scale = [int(grid_size / 2), int(grid_size / 2)] * state_dim
        return np.divide(data - shift, scale)

    def unnormalize(self, data):
        """Undo normalize.
        expects input data to be [sequence length, coordinates alternating between x and y]
        """
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        state_dim = data.shape[1] // 2

        x_shift = int(self.DEFAULT_GRID_SIZE / 2)
        y_shift = int(self.DEFAULT_GRID_SIZE / 2)
        x_scale = int(self.DEFAULT_GRID_SIZE / 2)
        y_scale = int(self.DEFAULT_GRID_SIZE / 2)

        data[:, ::2] = data[:, ::2] * x_scale + x_shift
        data[:, 1::2] = data[:, 1::2] * y_scale + y_shift

        return data.reshape(-1, state_dim * 2)

    @staticmethod
    def _unnormalize(data, grid_size):
        """Undo normalize.
        expects input data to be [sequence length, coordinates alternating between x and y]
        """
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        state_dim = data.shape[1] // 2

        x_shift = int(grid_size / 2)
        y_shift = int(grid_size / 2)
        x_scale = int(grid_size / 2)
        y_scale = int(grid_size / 2)

        data[:, ::2] = data[:, ::2] * x_scale + x_shift
        data[:, 1::2] = data[:, 1::2] * y_scale + y_shift

        return data.reshape(-1, state_dim * 2)

    def transform_to_centered_data(self, data, center_index):
        # implemented only for mice

        # data shape is seq_len, num_inds, num_kpts, kpts_dims -> seq_len*num_inds, num_kpts, kpts_dims
        data = data.reshape(-1, *data.shape[2:])

        # Center the data using given center_index
        mouse_center = data[:, center_index, :]
        centered_data = data - mouse_center[:, np.newaxis, :]

        # Rotate such that keypoints Tail base and neck are parallel with the y axis
        tail_base = self.BODY_PART_2_INDEX["tail_base"]
        neck = self.BODY_PART_2_INDEX["neck"]
        mouse_rotation = np.arctan2(
            data[:, tail_base, 0] - data[:, neck, 0],
            data[:, tail_base, 1] - data[:, neck, 1],
        )

        R = np.array(
            [
                [np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                [np.sin(mouse_rotation), np.cos(mouse_rotation)],
            ]
        ).transpose((2, 0, 1))

        # Encode mouse rotation as sine and cosine
        mouse_rotation = np.concatenate(
            [
                np.sin(mouse_rotation)[:, np.newaxis],
                np.cos(mouse_rotation)[:, np.newaxis],
            ],
            axis=-1,
        )

        centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
        centered_data = centered_data.transpose((0, 2, 1))

        centered_data = centered_data.reshape((-1, 24))

        # mean = np.mean(centered_data, axis=0)
        # centered_data = centered_data - mean
        return mouse_center, mouse_rotation, centered_data

    def transform_to_centeralign_components(self, data, center_index=7):

        seq_len, num_mice = data.shape[:2]

        mouse_center, mouse_rotation, centered_data = self.transform_to_centered_data(
            data, center_index
        )

        # Concatenate state as mouse center, mouse rotation and svd components
        data = np.concatenate([mouse_center, mouse_rotation, centered_data], axis=1)
        data = data.reshape(seq_len, num_mice, -1)

        return data

    def get_random_sample_from_sequence(self, sequence: np.ndarray):
        """
        Returns a training sample

        Randomly samples a section with length self.max_keypoints_len of the input sequence.
        """

        if self.interp_holes:
            sequence = self.fill_holes(sequence)

        start = np.random.randint(0, sequence.shape[0] - self.max_keypoints_len)
        end = start + self.max_keypoints_len
        keypoints = sequence[start:end, :]

        if self.augmentations:
            keypoints = keypoints.reshape(
                self.max_keypoints_len,
                self.NUM_INDIVIDUALS,
                self.NUM_KEYPOINTS,
                self.KPTS_DIMENSIONS,
            )
            keypoints = self.augmentations(keypoints)
            keypoints = keypoints.reshape(self.max_keypoints_len, -1)

        # Do scale, flatten and tensor AFTER features
        feats = self.featurise_keypoints(keypoints)

        # flatten for now
        feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)

        return feats

    def prepare_subsequence_sample(self, sequence: np.ndarray):
        """
        Returns a training sample
        """

        if self.augmentations:
            sequence = sequence.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            sequence = self.augmentations(sequence)
            sequence = sequence.reshape(self.max_keypoints_len, -1)

        feats = self.featurise_keypoints(sequence)

        # flatten for now
        feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)

        return feats

    def __getitem__(self, idx: int):

        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[
            subseq_ix[0], subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len
        ]
        inputs = self.prepare_subsequence_sample(subsequence)

        return inputs, []
