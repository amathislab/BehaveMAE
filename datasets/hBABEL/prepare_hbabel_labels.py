# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# For more details on our work, please refer to:
# Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders
# Lucas Stoffl, Andy Bonnetto, Stéphane d'Ascoli, Alexander Mathis
# https://www.biorxiv.org/content/10.1101/2024.08.06.606796v1
# --------------------------------------------------------

"""
## Generate hBABEL dataset
functions:
- `convert_babel_train_val` is a function to convert BABEL data in MABe22 format considering either train or val data,
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate hBABEL dataset from BABEL-teach"
    )
    parser.add_argument(
        "--babel_teach_path",
        type=str,
        help="Path to the BABEL-teach folder which contains train.json and val.json",
        default="./data/babel/babel-teach",
    )
    parser.add_argument(
        "--amass_pose_path",
        type=str,
        help="Path to the AMASS pose",
        default="./data/babel/babel-smplh-30fps-male",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/hBABEL",
        type=str,
        help="Path to save the hBABEL dataset",
    )

    args = parser.parse_args()
    return args


def get_joint_positions(data):
    """Get joint position from original data"""
    babel_ids = [dat["babel_id"] for dat in data]
    joint_positions = [dat["joint_positions_processed"] for dat in data]

    return joint_positions, babel_ids


def get_frame_number_maps(path_to_amass_pose, mode):
    """Create the frame number map which maps the labels
    and the videos using the frame numbers stated in the
    pose data"""

    path_to_pose = os.path.join(
        path_to_amass_pose, f"{mode}_proc_realigned_procrustes.pth.tar"
    )

    # Load data
    pose_data = joblib.load(path_to_pose)
    pose_data, babel_ids = get_joint_positions(pose_data)
    num_total_frames = np.sum([len(seq) for seq in pose_data])
    start = 0

    # Create frame number map
    frame_number_map = {}
    for seq_id, sequence_name in enumerate(babel_ids):
        joint_pos = pose_data[seq_id]
        end = start + joint_pos.shape[0]
        frame_number_map[sequence_name] = (start, end)
        start = end
    assert end == num_total_frames
    return frame_number_map


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def convert_babel(
    babel_teach_path: str,
    amass_pose_path: str,
    mode: str = "val",
    fps: int = 30,
    output_dir: str = "./data/hBABEL",
    train_size: float = 0.75,
    seed: int = 123,
    top_frame: int = 120,
    top_seq: int = 60,
):
    """Convert BABEL data in MABe format
    inputs:
        babel_teach_path: str: path to the BABEL data
        mode: str: "val" or "test"
        fps: int: frame per second
        output_dir: str: path to save the output files
        train_size: float: proportion of the data to use for training
        seed: int: random seed
        top_frame: int: number of top frame actions to keep
        top_seq: int: number of top sequence actions to keep
    """
    # Load data file
    os.makedirs(output_dir, exist_ok=True)
    for file_path in [
        os.path.join(babel_teach_path, path)
        for path in os.listdir(babel_teach_path)
        if path.endswith(".json")
    ]:
        if file_path.endswith(f"{mode}.json"):
            with open(file_path, "r") as f:
                data = json.load(f)

    # Go through every sequence to create a list of labels
    seq_recurrence = {}
    frame_recurrence = {}
    clip_length = {}
    seq_count = 0
    frame_number_map = get_frame_number_maps(amass_pose_path, mode)
    for sequence_name in frame_number_map.keys():
        seq_duration = (
            frame_number_map[sequence_name][1] - frame_number_map[sequence_name][0]
        )
        seq_count += seq_duration
        clip_length[sequence_name] = seq_duration
        # Get sequence labels
        sequence_labels = data[sequence_name]["seq_ann"]["labels"]
        for label_list in sequence_labels:
            if label_list["act_cat"] is not None:
                format_list = [
                    label.replace("/", "#") for label in label_list["act_cat"]
                ]
                for label in format_list:
                    seq_recurrence[label] = (
                        seq_recurrence[label] + 1
                        if label in seq_recurrence.keys()
                        else 1
                    )

        # Get frame labels
        if data[sequence_name]["frame_ann"] is not None:
            frame_labels = data[sequence_name]["frame_ann"]["labels"]
            for label_list in frame_labels:
                format_list = [
                    label.replace("/", "#") for label in label_list["act_cat"]
                ]
                for label in format_list:
                    frame_recurrence[label] = (
                        frame_recurrence[label] + 1
                        if label in frame_recurrence.keys()
                        else 1
                    )

    # Get most recurrent actions
    sorted_recurrence = sorted(frame_recurrence.items(), key=lambda x: x[1])
    sorted_recurrence_seg = sorted(seq_recurrence.items(), key=lambda x: x[1])
    vocabulary = [action[0] + "_seg" for action in sorted_recurrence_seg[-top_seq:]] + [
        action[0] + "_frame" for action in sorted_recurrence[-top_frame:]
    ]
    label_array = np.zeros((len(vocabulary), seq_count))
    empty_episodes = []
    for sequence_name, (start_sequence, stop_sequence) in frame_number_map.items():
        if data[sequence_name]["frame_ann"] is not None:
            # Get frame labels
            frame_labels = data[sequence_name]["frame_ann"]["labels"]
            for label_list in frame_labels:
                format_list = [
                    label.replace("/", "#") for label in label_list["act_cat"]
                ]
                start_frame = round(label_list["start_t"] * fps)
                end_frame = round(label_list["end_t"] * fps)
                for label in format_list:
                    if label + "_frame" in vocabulary:
                        action_ind = vocabulary.index(label + "_frame")
                        label_array[
                            action_ind,
                            start_frame + start_sequence : end_frame + start_sequence,
                        ] = 1
            # Get sequence labels
            sequence_labels = data[sequence_name]["seq_ann"]["labels"]
            for label_list in sequence_labels:
                if label_list["act_cat"] is not None:
                    format_list = [
                        label.replace("/", "#") for label in label_list["act_cat"]
                    ]
                    for label in format_list:
                        if label + "_seg" in vocabulary:
                            action_ind = vocabulary.index(label + "_seg")
                            label_array[action_ind, start_sequence:stop_sequence] = 1
                else:
                    empty_episodes.append(sequence_name)
                    label_array[:, start_sequence:stop_sequence] = -1

        else:
            empty_episodes.append(sequence_name)
            label_array[:, start_sequence:stop_sequence] = -1

    frame_number_map = {
        key: value
        for key, value in frame_number_map.items()
        if key not in empty_episodes
    }
    task_types = ["Discrete"] * len(vocabulary)
    data_dict = {
        "frame_number_map": frame_number_map,
        "label_array": label_array,
        "vocabulary": vocabulary,
        "task_type": task_types,
    }
    np.save(
        os.path.join(
            output_dir,
            f"hbabel_val_test_actions_{mode}_top_{top_frame}_{top_seq}_filtered.npy",
        ),
        data_dict,
        allow_pickle=True,
    )

    # Save new frame_number_map
    np.save(
        os.path.join(output_dir, f"frame_number_map_{mode}_filtered.npy"),
        frame_number_map,
        allow_pickle=True,
    )

    # Create Split info file
    train_seq, test_seq = train_test_split(
        list(frame_number_map.keys()), train_size=train_size, random_state=seed
    )
    split_info_dict = {
        "SubmissionTrain": train_seq,
        "publicTest": test_seq,
        "privateTest": test_seq,
    }
    save_json(
        split_info_dict,
        os.path.join(output_dir, f"split_info_hBABEL_{mode}_filtered.json"),
    )
    # Create Tasks info file
    task_info_dict = {
        "seeds": [41, 42, 43],
        "task_id_list": vocabulary,
        "sequence_level_tasks": ["There_are_no_sequence_task_here"],
    }
    save_json(
        task_info_dict,
        os.path.join(
            output_dir, f"task_info_hBABEL_{mode}_top_{top_frame}_{top_seq}.json"
        ),
    )
    # Create clip length file
    save_json(clip_length, os.path.join(output_dir, f"clip_length_hBABEL_{mode}.json"))


if __name__ == "__main__":
    # python datasets/hBABEL/prepare_hbabel_labels.py

    args = parse_args()
    path_to_babel_teach = args.babel_teach_path
    output_dir = args.output_dir
    amass_pose_path = args.amass_pose_path

    print("Generating train data...")
    convert_babel(
        path_to_babel_teach,
        amass_pose_path,
        mode="train",
        output_dir=output_dir,
        top_frame=120,
        top_seq=60,
    )
    print("Generating val data...")
    convert_babel(
        path_to_babel_teach,
        amass_pose_path,
        mode="val",
        output_dir=output_dir,
        top_frame=120,
        top_seq=60,
    )
