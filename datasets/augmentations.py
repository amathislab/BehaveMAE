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

import random

import numpy as np

# Some functions are adapted from TREBA
# "TREBA" by Sun, Jennifer J and Kennedy, Ann and Zhan, Eric and Anderson, David J and Yue, Yisong and Perona, Pietro is licensed under CC BY-NC-SA 4.0 license.
# https://github.com/neuroethology/TREBA/blob/c522e169738f5225298cd4577e5df9085130ce8a/util/datasets/mouse_v1/augmentations/augmentation_functions.py


class GaussianNoise:
    def __init__(self, p=0.5, mu=0, sigma=2) -> None:
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            noise = np.random.normal(self.mu, self.sigma, keypoints.shape).astype(
                np.float32
            )
            noisy_kpts = keypoints.copy()
            noisy_kpts = noisy_kpts + noise
            return noisy_kpts

        return keypoints


class Rotation:
    def __init__(self, grid_size, p=0.5, rotation_range=np.pi) -> None:
        self.p = p
        self.rotation_range = rotation_range
        self.grid_size = grid_size

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return keypoints

        original = keypoints.copy()
        rot_kpts = keypoints.copy()
        rot_kpts = rot_kpts.transpose(
            1, 0, 2, 3
        )  # => should transpose to (NUM_MICE, seq_len, num_bpts, 2)

        image_center = [self.grid_size[0] / 2, self.grid_size[1] / 2]

        mouse_rotation = np.repeat(
            np.random.uniform(low=-1 * self.rotation_range, high=self.rotation_range),
            rot_kpts.shape[1],
        )
        R = np.array(
            [
                [np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                [np.sin(mouse_rotation), np.cos(mouse_rotation)],
            ]
        ).transpose((2, 0, 1))

        # iterate over number of animals/individuals
        for i in range(len(rot_kpts)):
            rot_kpts[i] = (
                np.matmul(R, (rot_kpts[i] - image_center).transpose(0, 2, 1)).transpose(
                    0, 2, 1
                )
                + image_center
            )

        # Check if possible for trajectory to fit within borders
        bounded = (
            (np.amax(rot_kpts[:, :, :, 0]) - np.amin(rot_kpts[:, :, :, 0]))
            < self.grid_size[0]
        ) and (
            (np.amax(rot_kpts[:, :, :, 1]) - np.amin(rot_kpts[:, :, :, 1]))
            < self.grid_size[1]
        )

        if bounded:
            # Shift all points to within borders first
            horizontal_shift = np.amax(rot_kpts[:, :, :, 0] - self.grid_size[0])
            horizontal_shift_2 = np.amin(rot_kpts[:, :, :, 0])
            if horizontal_shift > 0:
                rot_kpts[:, :, :, 0] = rot_kpts[:, :, :, 0] - horizontal_shift
            if horizontal_shift_2 < 0:
                rot_kpts[:, :, :, 0] = rot_kpts[:, :, :, 0] - horizontal_shift_2

            vertical_shift = np.amax(rot_kpts[:, :, :, 1] - self.grid_size[1])
            vertical_shift_2 = np.amin(rot_kpts[:, :, :, 1])
            if vertical_shift > 0:
                rot_kpts[:, :, :, 1] = rot_kpts[:, :, :, 1] - vertical_shift
            if vertical_shift_2 < 0:
                rot_kpts[:, :, :, 1] = rot_kpts[:, :, :, 1] - vertical_shift_2

            max_horizontal_shift = np.amin(self.grid_size[0] - rot_kpts[:, :, :, 0])
            min_horizontal_shift = np.amin(rot_kpts[:, :, :, 0])
            max_vertical_shift = np.amin(self.grid_size[1] - rot_kpts[:, :, :, 1])
            min_vertical_shift = np.amin(rot_kpts[:, :, :, 1])
            horizontal_shift = np.random.uniform(
                low=-1 * min_horizontal_shift, high=max_horizontal_shift
            )
            vertical_shift = np.random.uniform(
                low=-1 * min_vertical_shift, high=max_vertical_shift
            )

            rot_kpts[:, :, :, 0] = rot_kpts[:, :, :, 0] + horizontal_shift
            rot_kpts[:, :, :, 1] = rot_kpts[:, :, :, 1] + vertical_shift

            rot_kpts = rot_kpts.transpose(1, 0, 2, 3)

            return rot_kpts
        else:
            return original


class Reflect:
    def __init__(self, grid_size, p=0.5) -> None:
        self.p = p
        self.grid_size = grid_size

    def reflect_points(self, points, A, B, C):
        # A * x + B * y + C = 0
        new_points = np.zeros(points.shape)

        M = np.sqrt(A * A + B * B)
        A = A / M
        B = B / M
        C = C / M

        D = A * points[:, :, :, 0] + B * points[:, :, :, 1] + C

        new_points[:, :, :, 0] = points[:, :, :, 0] - 2 * A * D
        new_points[:, :, :, 1] = points[:, :, :, 1] - 2 * B * D

        return new_points

    def __call__(self, keypoints):
        if np.random.random() > self.p:
            return keypoints

        if np.random.random() > 0.5:
            new_keypoints = self.reflect_points(
                keypoints, 0, 1, -self.grid_size[1] // 2
            )
        else:
            new_keypoints = self.reflect_points(
                keypoints, 1, 0, -self.grid_size[0] // 2
            )
        return new_keypoints


# Adapted from BABEL:
# https://github.com/abhinanda-punnakkal/BABEL/blob/main/action_recognition/feeders/tools.py


class RandomMove:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def random_move(
        self,
        data_numpy,
        angle_candidate=[-10.0, -5.0, 0.0, 5.0, 10.0],
        scale_candidate=[0.9, 1.0, 1.1],
        transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
        move_time_candidate=[1],
    ):
        data_numpy = data_numpy.transpose(2, 0, 1)  # C, T, V
        # input: C,T,V
        C, T, V = data_numpy.shape
        move_time = random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i] : node[i + 1]] = (
                np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            )
            s[node[i] : node[i + 1]] = np.linspace(
                S[i], S[i + 1], node[i + 1] - node[i]
            )
            t_x[node[i] : node[i + 1]] = np.linspace(
                T_x[i], T_x[i + 1], node[i + 1] - node[i]
            )
            t_y[node[i] : node[i + 1]] = np.linspace(
                T_y[i], T_y[i + 1], node[i + 1] - node[i]
            )

        theta = np.array(
            [[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]]
        )

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :] = new_xy.reshape(2, V)

        return data_numpy.transpose(1, 2, 0)

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return self.random_move(keypoints)
        return keypoints


class RandomShift:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def random_shift(self, data_numpy):
        data_numpy = data_numpy.transpose(2, 0, 1)  # C, T, V
        # input: C,T,V
        C, T, V = data_numpy.shape
        data_shift = np.zeros(data_numpy.shape)
        valid_frame = (data_numpy != 0).sum(axis=2).sum(axis=0) > 0
        begin = valid_frame.argmax()
        end = len(valid_frame) - valid_frame[::-1].argmax()

        size = end - begin
        bias = random.randint(0, T - size)
        data_shift[:, bias : bias + size, :] = data_numpy[:, begin:end, :]

        return data_shift.transpose(1, 2, 0)

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return self.random_shift(keypoints)
        return keypoints
