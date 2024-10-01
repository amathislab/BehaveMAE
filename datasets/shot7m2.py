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

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from .augmentations import GaussianNoise
from .pose_traj_dataset import BasePoseTrajDataset


class SHOT7M2Dataset(BasePoseTrajDataset):
    """
    Synthetic Hierarchical and cOmpositional baskeTball dataset
    """

    DEFAULT_FRAME_RATE = 30
    NUM_KEYPOINTS = 26
    KPTS_DIMENSIONS = 3
    NUM_INDIVIDUALS = 1
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    SAMPLE_LEN = 1800

    STR_BODY_PARTS = [
        "center",
        "l_hip",
        "l_knee",
        "l_ankle",
        "l_foot",
        "l_toes",
        "r_hip",
        "r_knee",
        "r_ankle",
        "r_foot",
        "r_toes",
        "lumbars",
        "low_thorax",
        "high_thorax",
        "cervicals",
        "l_shoulder_blade",
        "l_shoulder",
        "l_elbow",
        "l_wrist",
        "neck",
        "head",
        "head_top",
        "r_shoulder_blade",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
    ]

    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

    SPLIT_INDS = [
        0, 1, 6, 11,  # center,l_hip,r_hip,lumbars
        2, 3, 4, 5,  # left leg
        7, 8, 9, 10,  # right leg
        14, 19, 20, 21,  # cervicals, neck, head, head_top
        15, 16, 17, 18,  # left arm
        22, 23, 24, 25,  # right arm
    ]

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        augmentations: transforms.Compose = None,
        include_testdata: bool = False,
        split_tokenization: bool = False,
        **kwargs
    ):
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE  # downsample frames if needed

        self.mode = mode

        if augmentations:
            self.augmentations = transforms.Compose(
                [
                    GaussianNoise(p=0.5),
                ]
            )
        else:
            self.augmentations = None

        self.load_data(include_testdata)

        self.split_tokenization = split_tokenization

        self.preprocess()

    def load_data(self, include_testdata) -> None:
        """Loads dataset"""
        if self.mode == "pretrain":
            self.raw_data = np.load(self.path, allow_pickle=True).item()
            if include_testdata:
                raw_data_test = np.load(
                    self.path.replace("train", "test"), allow_pickle=True
                ).item()
                self.raw_data["sequences"].update(raw_data_test["sequences"])
        elif self.mode == "test":
            self.raw_data = np.load(
                self.path.replace("train", "test"), allow_pickle=True
            ).item()
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def featurise_keypoints(self, keypoints):
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints

    def preprocess(self):
        """
        Does initial preprocessing on entire dataset.
        """
        sequences = self.raw_data["sequences"]["keypoints"]

        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window

        for seq_ix, vec_seq in enumerate(sequences.values()):
            # Preprocess sequences

            if self.split_tokenization:
                vec_seq = vec_seq[:, :, self.SPLIT_INDS]
                self.NUM_KEYPOINTS = self.NUM_KEYPOINTS - 2
                self.KEYFRAME_SHAPE = (
                    self.NUM_INDIVIDUALS,
                    self.NUM_KEYPOINTS,
                    self.KPTS_DIMENSIONS,
                )

            vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)

            # Pads the beginning and end of the sequence with duplicate frames
            if sub_seq_length < 120:
                pad_length = sub_seq_length
            else:
                pad_length = 120
            pad_vec = np.pad(
                vec_seq,
                ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                mode="edge",
            )

            seq_keypoints.append(pad_vec)

            keypoints_ids.extend(
                [
                    (seq_ix, i)
                    for i in np.arange(
                        0, len(pad_vec) - sub_seq_length + 1, sliding_window
                    )
                ]
            )

        seq_keypoints = np.array(seq_keypoints, dtype=np.float32)

        self.items = list(np.arange(len(keypoints_ids)))

        self.seq_keypoints = seq_keypoints
        self.keypoints_ids = keypoints_ids
        self.n_frames = len(self.keypoints_ids)

        del self.raw_data

    @staticmethod
    def fill_holes():
        pass

    def normalize(self):
        pass

    def unnormalize(self):
        pass

    def transform_to_centered_data(self):
        pass

    def transform_to_centeralign_components(self):
        pass

    def prepare_subsequence_sample(self, sequence: np.ndarray):
        """
        Returns a training sample
        """

        if self.augmentations:
            sequence = sequence.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            sequence = self.augmentations(sequence)
            sequence = sequence.reshape(self.max_keypoints_len, -1)

        feats = self.featurise_keypoints(sequence)

        feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)

        return feats

    def __getitem__(self, idx: int):

        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[
            subseq_ix[0], subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len
        ]
        inputs = self.prepare_subsequence_sample(subsequence)

        return inputs, []
