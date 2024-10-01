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

import joblib
import numpy as np
import torch
from tqdm import tqdm

from ..hbabel import hBABELDataset


def procrustes_alignment(
    trial_data: np.ndarray, reference_pose: np.ndarray
) -> np.ndarray:
    """
    Perform Procrustes alignment on a single trial.

    Parameters:
    - trial_data: 3D numpy array representing the keypoints of multiple_trials trial.
    - reference_pose: 1D numpy array representing the reference pose.

    Returns:
    - aligned_trial: Aligned keypoints for a single trial.
    """
    aligned_data = np.zeros_like(trial_data)

    # Centering
    center_trial = trial_data[0] - np.mean(trial_data[0], axis=0)
    center_reference = reference_pose - np.mean(reference_pose, axis=0)

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(np.dot(center_trial.T, center_reference))

    # Rotation matrix
    R = np.dot(U, Vt)

    # Apply rotation and translation to align with the reference pose
    for t in range(trial_data.shape[0]):
        center_trial = trial_data[t] - np.mean(trial_data[t], axis=0)
        aligned_data[t] = np.dot(center_trial, R) + np.mean(reference_pose, axis=0)

    return aligned_data


# to compute hBABEL stats (mean & std) for normalization of poses


def compute_stats_feats(feats: torch.Tensor):
    feats = torch.cat(feats)
    mean = feats.mean(0)
    std = feats.std(0)
    return mean, std


def _save_processed_ntu_data() -> None:

    for ds in ["train", "val", "test"]:
        print(f"Reproject for {ds}")
        data = joblib.load(f"data/babel/babel-smplh-30fps-male/{ds}.pth.tar")

        for sample in tqdm(data):

            sequence = sample["joint_positions"]
            features = sequence[:, hBABELDataset.NTU_KPTS, :]
            features = features.transpose(2, 0, 1)[:, :, :, np.newaxis]
            # Normalize (pre-process) in NTU RGBD-style
            features = hBABELDataset.ntu_pre_normalization(features)
            features = features.transpose(1, 2, 3, 0).squeeze()

            features = features.reshape(features.shape[0], -1)
            # features = hBABELDataset._normalize(features)

            sample["joint_positions_processed"] = features.reshape(
                features.shape[0], -1, 3
            ).astype(np.float32)

        print(f"Apply procrustes for {ds}")
        reference_pose = np.mean(
            [sample["joint_positions_processed"][0] for sample in data], axis=0
        )
        for sample in tqdm(data):
            sample["joint_positions_processed"] = procrustes_alignment(
                sample["joint_positions_processed"], reference_pose
            )

        joblib.dump(
            data,
            f"data/babel/babel-smplh-30fps-male/{ds}_proc_realigned_procrustes.pth.tar",
        )


if __name__ == "__main__":

    # to be ran with: python -m datasets.hBABEL.prepare_hbabel_data

    _save_processed_ntu_data()
