#
# Copyright 2024-present by A. Mathis Group and contributors. All rights reserved.
#
"""
## Generate hBABEL dataset
functions:
- `convert_babel_train_val` is a function to convert BABEL data in MABe22 format considering either train or val data,
"""

import argparse
import os
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate hBABEL dataset from BABEL-teach"
    )
    parser.add_argument(
        "--babel_teach_path",
        type=str,
        help="Path to the BABEL-teach folder which contains train.json and val.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/dataset",
        type=str,
        help="Path to save the hBABEL dataset",
    )

    args = parser.parse_args()
    return args


def convert_babel_train_val(
    babel_teach_path: str,
    output_dir: str,
    train_on: str = "train",
    fps: int = 30,
    top_frame: int = 120,
    top_seq: int = 60,
    seeds: list = [41, 42, 43],
):
    """Convert BABEL data in MABe format considering both train and val data
    inputs:
        babel_teach_path: str: path to the BABEL data
        train_on: str: "train" or "val"
        fps: int: frame per second
        output_dir: str: path to save the output files
        top_frame: int: number of top frame actions to keep
        top_seq: int: number of top sequence actions to keep
        seeds: list: list of seeds to use
    """

    # Check input
    assert isinstance(seeds, list)
    assert train_on in ["train", "val"]

    # Load data file
    os.makedirs(output_dir, exist_ok=True)
    train_data, val_data = None, None
    for file_path in [
        os.path.join(babel_teach_path, path)
        for path in os.listdir(babel_teach_path)
        if path.endswith(".json")
    ]:
        if file_path.endswith(f"train.json"):
            with open(file_path, "r") as f:
                train_data = json.load(f)
        elif file_path.endswith(f"val.json"):
            with open(file_path, "r") as f:
                val_data = json.load(f)

    if train_data is None:
        raise FileNotFoundError(f"Train data not found in {babel_teach_path}")
    if val_data is None:
        raise FileNotFoundError(f"Val data not found in {babel_teach_path}")

    # Go through every sequence to create a list of labels
    seq_recurrence = {}
    frame_recurrence = {}
    clip_length = {}
    seq_count = 0
    frame_number_map_train = np.load(
        "./hBABEL/frame_number_map_train.npy",
        allow_pickle=True,
    ).item()
    frame_number_map_val = np.load(
        "./hBABEL/frame_number_map_val.npy",
        allow_pickle=True,
    ).item()
    empty_episodes = []

    frame_number_map = frame_number_map_train.copy()
    frame_number_map.update(frame_number_map_val)

    new_frame_number_map = {}
    # Get most recurrent actions
    for sequence_name in frame_number_map.keys():
        data = train_data if sequence_name in train_data.keys() else val_data
        # Get sequence labels
        sequence_labels = data[sequence_name]["seq_ann"]["labels"]
        empty_episodes = []
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
            else:
                empty_episodes.append(sequence_name)
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

        else:
            empty_episodes.append(sequence_name)

        if sequence_name not in empty_episodes:
            seq_duration = (
                frame_number_map[sequence_name][1] - frame_number_map[sequence_name][0]
            )
            new_frame_number_map[sequence_name] = (seq_count, seq_count + seq_duration)
            seq_count += seq_duration
            clip_length[sequence_name] = seq_duration

    sorted_recurrence = sorted(frame_recurrence.items(), key=lambda x: x[1])
    sorted_recurrence_seg = sorted(seq_recurrence.items(), key=lambda x: x[1])
    vocabulary = [action[0] + "_seg" for action in sorted_recurrence_seg[-top_seq:]] + [
        action[0] + "_frame" for action in sorted_recurrence[-top_frame:]
    ]
    label_array = np.zeros((len(vocabulary), seq_count))
    # Get label array
    seq_count = 0
    for sequence_name, (start_sequence, stop_sequence) in new_frame_number_map.items():
        data = train_data if sequence_name in train_data.keys() else val_data
        assert data[sequence_name]["frame_ann"] is not None
        seq_duration = stop_sequence - start_sequence
        # Get frame labels
        frame_labels = data[sequence_name]["frame_ann"]["labels"]
        for label_list in frame_labels:
            format_list = [label.replace("/", "#") for label in label_list["act_cat"]]
            start_frame = round(label_list["start_t"] * fps)
            end_frame = round(label_list["end_t"] * fps)
            for label in format_list:
                if label + "_frame" in vocabulary:
                    action_ind = vocabulary.index(label + "_frame")
                    label_array[
                        action_ind, seq_count + start_frame : seq_count + end_frame
                    ] = 1
        # Get sequence labels
        sequence_labels = data[sequence_name]["seq_ann"]["labels"]
        for label_list in sequence_labels:
            assert label_list["act_cat"] is not None
            format_list = [label.replace("/", "#") for label in label_list["act_cat"]]
            for label in format_list:
                if label + "_seg" in vocabulary:
                    action_ind = vocabulary.index(label + "_seg")
                    label_array[action_ind, seq_count : seq_count + seq_duration] = 1

        seq_count += seq_duration

    # Save Label file
    task_types = ["Discrete"] * (len(vocabulary))
    data_dict = {
        "frame_number_map": new_frame_number_map,
        "label_array": label_array,
        "vocabulary": vocabulary,
        "task_type": task_types,
    }
    np.save(
        os.path.join(
            output_dir, f"babel_trainval_actions_top_{top_frame}_{top_seq}_filtered.npy"
        ),
        data_dict,
        allow_pickle=True,
    )

    # Save new frame_number_map
    np.save(
        os.path.join(output_dir, f"frame_number_map_trainval_filtered.npy"),
        new_frame_number_map,
        allow_pickle=True,
    )

    # Create Split info file
    train_seq = list(
        set(frame_number_map_train.keys()) & set(new_frame_number_map.keys())
    )
    test_seq = list(set(frame_number_map_val.keys()) & set(new_frame_number_map.keys()))
    if train_on == "train":
        split_info_dict = {
            "SubmissionTrain": train_seq,
            "publicTest": test_seq,
            "privateTest": test_seq,
        }
    elif train_on == "val":
        split_info_dict = {
            "SubmissionTrain": test_seq,
            "publicTest": train_seq,
            "privateTest": train_seq,
        }
    else:
        raise NotImplementedError()

    with open(
        os.path.join(output_dir, f"split_info_BABEL_trainval_{train_on}_filtered.json"),
        "w",
    ) as f:
        json.dump(split_info_dict, f)

    # Create Tasks info file
    task_info_dict = {
        "seeds": seeds,
        "task_id_list": vocabulary,
        "sequence_level_tasks": ["There_are_no_sequence_task_here"],
    }
    single_seed = "" if len(seeds) > 1 else "_singleseed"
    with open(
        os.path.join(
            output_dir,
            f"task_info_BABEL_trainval_top_{top_frame}_{top_seq}{single_seed}.json",
        ),
        "w",
    ) as f:
        json.dump(task_info_dict, f)

    # Create clip length file
    with open(os.path.join(output_dir, f"clip_length_BABEL_trainval.json"), "w") as f:
        json.dump(clip_length, f)


if __name__ == "__main__":
    # python hBABEL/generate_hbabel.py

    args = parse_args()
    path_to_babel_teach = args.babel_teach_path
    output_dir = args.output_dir
    print("Generating train data...")
    convert_babel_train_val(
        path_to_babel_teach, output_dir, train_on="train", seeds=[42]
    )
    print("Generating val data...")
    convert_babel_train_val(path_to_babel_teach, output_dir, train_on="val", seeds=[42])
