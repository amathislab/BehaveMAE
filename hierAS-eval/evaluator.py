import argparse
import os
import random
from math import floor, log10

import numpy as np
import pandas as pd
from eval.training import train_and_eval

hash_code = str(random.getrandbits(16))
TRY_NAME = f"TRY_{hash_code}"


class Paths:
    SUBMISSION_DATA_PATH = os.getenv(
        "SUBMISSION_DATA_PATH", "./example_data/example_embeddings.npy"
    )
    LABELS_PATH = os.getenv("LABELS_PATH", "./example_data/example_labels.npy")
    SPLIT_INFO_FILE = os.getenv("SPLIT_INFO_FILE", "./example_data/example_split.json")
    TASK_INFO_FILE = os.getenv("TASK_INFO_FILE", "eval/metadata/jax_tasks.json")
    CLIP_LENGTHS_FILE = os.getenv(
        "CLIP_LENGTHS_FILE", "eval/metadata/clip_lengths_mouse_triplets.json"
    )
    FRAME_NUMBER_MAP = os.getenv(
        "FRAME_MAP_FILE", "eval/metadata/mouse_frame_number_map.npy"
    )
    os.makedirs("./temp", exist_ok=True)
    LOG_PATH = os.getenv("TRAINING_LOG_PATH", f"./temp/{TRY_NAME}")
    SHORT_RUN = False  # For fast testing


def round_sig(x, sig=4):
    if not x == 0:
        return round(x, sig - int(floor(log10(abs(x)))) - 1)
    else:
        return x


def validate_submission(submission_file_path, embedding_max_size, frame_map_file):
    submission = np.load(submission_file_path, allow_pickle=True).item()

    if not isinstance(submission, dict):
        raise ValueError("Submission should be dict")

    frame_map = np.load(frame_map_file, allow_pickle=True).item()

    if "frame_number_map" not in submission:
        raise ValueError("Frame number map missing")

    for k, v in frame_map.items():
        sv = submission["frame_number_map"][k]
        if not v == sv:
            raise ValueError(
                "Frame number map should be exactly same as provided in frame_number_map.npy in resources"
            )

    if "embeddings" not in submission:
        raise ValueError("Embeddings array missing")
    elif not isinstance(submission["embeddings"], np.ndarray):
        raise ValueError("Embeddings should be a numpy array")
    elif not len(submission["embeddings"].shape) == 2:
        raise ValueError("Embeddings should be 2D array")
    elif not submission["embeddings"].shape[1] <= embedding_max_size:
        raise ValueError(f"Embeddings too large, max allowed is {embedding_max_size}")
    elif not isinstance(submission["embeddings"][0, 0], np.float32):
        raise ValueError(f"Embeddings are not float32")

    if not np.isfinite(submission["embeddings"]).all():
        raise ValueError("Emebddings contains NaN or infinity")

    print("All checks passed")
    del submission


class AIcrowdEvaluator:
    def __init__(
        self,
        ground_truth_path,
        task_name="flies",
        training_size="75",
        threshold=0.5,
        apply_filter=False,
        **kwargs,
    ):
        Paths.LABELS_PATH = ground_truth_path
        self.task_name = task_name
        self.training_size = training_size
        self.threshold = threshold
        self.apply_filter = apply_filter

        print("Starting evaluation task", task_name)

    def get_results(self):
        results_df = pd.read_csv(os.path.join(Paths.LOG_PATH, "results.csv"))
        return results_df

    def _evaluate(self, client_payload, _context={}):
        submission_file_path = client_payload["submission_file_path"]
        Paths.SUBMISSION_DATA_PATH = submission_file_path

        if not os.path.exists(Paths.LOG_PATH):
            os.mkdir(Paths.LOG_PATH)
        if self.task_name == "mabe_mice":
            Paths.TASK_INFO_FILE = "eval/metadata/jax_tasks.json"
            Paths.SPLIT_INFO_FILE = "eval/metadata/jax_split.json"
            Paths.CLIP_LENGTHS_FILE = os.getenv(
                "CLIP_LENGTHS_FILE", "eval/metadata/clip_lengths_mouse_triplets.json"
            )
            Paths.FRAME_NUMBER_MAP = os.getenv(
                "FRAME_MAP_FILE", "eval/metadata/mouse_frame_number_map.npy"
            )
            embedding_max_size = 128
            test_size = 0.1

        elif "Shot" in self.task_name:
            Paths.TASK_INFO_FILE = (
                f"eval/metadata/{self.task_name}/task_info_{self.task_name}.json"
            )
            Paths.SPLIT_INFO_FILE = f"eval/metadata/{self.task_name}/split_info_Shot_{self.training_size}_{self.task_name}.json"
            Paths.CLIP_LENGTHS_FILE = os.getenv(
                "CLIP_LENGTHS_FILE",
                f"eval/metadata/{self.task_name}/clip_lengths_{self.task_name}.json",
            )
            Paths.FRAME_NUMBER_MAP = os.getenv(
                "FRAME_MAP_FILE",
                f"eval/metadata/{self.task_name}/frame_number_map_{self.task_name}.npy",
            )
            embedding_max_size = 64
            # test_size = 0.25
            test_size = 1 - int(self.training_size) / 100

        elif self.task_name == "hBABEL":
            Paths.TASK_INFO_FILE = (
                f"eval/metadata/{self.task_name}/task_info_hBABEL_val_top_120_60.json"
            )
            Paths.SPLIT_INFO_FILE = (
                f"eval/metadata/{self.task_name}/split_info_hBABEL_val_filtered.json"
            )

            Paths.CLIP_LENGTHS_FILE = (
                f"eval/metadata/{self.task_name}/clip_length_hBABEL_val_filtered.json"
            )
            Paths.FRAME_NUMBER_MAP = (
                f"eval/metadata/{self.task_name}/frame_number_map_val_filtered.npy"
            )
            embedding_max_size = 64
            # embedding_max_size = 128
            test_size = 0.1

        print(Paths.CLIP_LENGTHS_FILE)
        validate_submission(
            Paths.SUBMISSION_DATA_PATH,
            embedding_max_size,
            Paths.FRAME_NUMBER_MAP,
        )

        print("Starting training")
        train_and_eval.run_all_tasks(
            Paths,
            test_size,
            self.task_name,
            threshold=self.threshold,
            apply_filter=self.apply_filter,
        )

        results = self.get_results()

        return results


def results_summary_mice(path, sub_nr=""):

    res = pd.read_csv(path)

    scores = [f"Score Alpha {x}" for x in [0.1, 0.5, 1.0, 2.0, 5.0]]

    print("Submission at hierarchy level:", sub_nr)

    print("All MSE", "\t\t", "All F1", "\t\t", "Frame F1", "\t\t", "Sequence F1")
    all_mse = round(res.loc[res["Metric"] == "mean_mse", "Private Score"].mean(), 4)
    all_mse_std = round(
        res.loc[res["Metric"] == "mean_mse", scores].to_numpy().std(axis=1).mean(), 4
    )
    all_f1 = (round(res.loc[res["Metric"] == "f1_score", "Private Score"].mean(), 3),)
    all_f1_std = (
        round(
            res.loc[res["Metric"] == "f1_score", scores].to_numpy().std(axis=1).mean(),
            3,
        ),
    )
    frame_f1 = round(
        res.loc[
            (res["Metric"] == "f1_score") & (res["Sequence Level Task"] == False),
            "Private Score",
        ].mean(),
        3,
    )
    frame_f1_std = round(
        res.loc[
            (res["Metric"] == "f1_score") & (res["Sequence Level Task"] == False),
            scores,
        ]
        .to_numpy()
        .std(axis=1)
        .mean(),
        3,
    )
    seq_f1 = round(
        res.loc[
            (res["Metric"] == "f1_score") & (res["Sequence Level Task"] == True),
            "Private Score",
        ].mean(),
        3,
    )
    seq_f1_std = round(
        res.loc[
            (res["Metric"] == "f1_score") & (res["Sequence Level Task"] == True), scores
        ]
        .to_numpy()
        .std(axis=1)
        .mean(),
        3,
    )
    print(
        all_mse,
        "\u00B1",
        all_mse_std,
        "\t",
        all_f1[0],
        "\u00B1",
        all_f1_std[0],
        "\t\t",
        frame_f1,
        "\u00B1",
        frame_f1_std,
        "\t\t",
        seq_f1,
        "\u00B1",
        seq_f1_std,
    )

    print("\n")

    return (all_mse, all_f1, frame_f1, seq_f1)


def results_summary_shot7m2(path, sub_nr, filter_activity_player=False):

    res = pd.read_csv(path)

    scores = [f"Score Alpha {x}" for x in [0.1, 0.5, 1.0, 2.0, 5.0]]

    def compute_our_movemes_mean(scores):
        scores1 = scores[:6]
        scores2 = scores[6:].reshape(-1, 2).mean(axis=1)
        scores_ = np.concatenate([scores1, scores2])
        return scores_.mean()

    def compute_our_actions_mean(scores):
        scores1 = scores[:6]
        scores2 = scores[8:10]
        scores3 = scores[6:8].reshape(-1, 2).mean(axis=1)
        scores4 = scores[10:].reshape(-1, 2).mean(axis=1)
        scores_ = np.concatenate([scores1, scores2, scores3, scores4])
        return scores_.mean()

    print("Submission at hierarchy level:", sub_nr)

    print("All F1", "\t\t")
    all_f1 = round(res.loc[res["Metric"] == "f1_score", "Private Score"].mean(), 3)
    all_f1_std = (
        round(
            res.loc[res["Metric"] == "f1_score", scores].to_numpy().std(axis=1).mean(),
            3,
        ),
    )
    print(all_f1, "\u00B1", all_f1_std, "\t\t")

    print("Activity F1", "\t\t", "Action F1", "\t\t", "Moveme F1")
    activity_ind = [
        i
        for i in range(len(res.index))
        if res["Task ID"].to_list()[i].startswith("activity")
    ]
    if filter_activity_player:
        activity_ind = [
            i
            for i in range(len(res.index))
            if res["Task ID"].to_list()[i].startswith("activity_Episode")
        ]
    activity_f1 = round(res.loc[activity_ind, "Private Score"].mean(), 3)
    activity_f1_std = round(
        res.loc[activity_ind, scores].to_numpy().std(axis=1).mean(), 3
    )

    action_ind = [
        i
        for i in range(len(res.index))
        if res["Task ID"].to_list()[i].startswith("action")
    ]
    action_f1 = round(res.loc[action_ind, "Private Score"].mean(), 3)
    action_f1_std = round(res.loc[action_ind, scores].to_numpy().std(axis=1).mean(), 3)

    moveme_ind = [
        i
        for i in range(len(res.index))
        if res["Task ID"].to_list()[i].startswith("moveme")
    ]
    moveme_f1 = round(res.loc[moveme_ind, "Private Score"].mean(), 3)
    moveme_f1_std = round(res.loc[moveme_ind, scores].to_numpy().std(axis=1).mean(), 3)

    print(
        activity_f1,
        "\u00B1",
        activity_f1_std,
        "\t\t",
        action_f1,
        "\u00B1",
        action_f1_std,
        "\t\t",
        moveme_f1,
        "\u00B1",
        moveme_f1_std,
        "\t\t",
    )

    our_action_f1 = compute_our_actions_mean(
        res[res["Task ID"].str.startswith("action")]["Private Score"].to_numpy()
    )
    our_moveme_f1 = compute_our_movemes_mean(
        res[res["Task ID"].str.startswith("moveme")]["Private Score"].to_numpy()
    )
    our_all_f1 = np.mean([activity_f1, our_action_f1, our_moveme_f1])

    print(
        "\n",
        "Our All F1:",
        our_all_f1,
        "\n",
        "Our Activity F1:",
        activity_f1,
        "\n",
        "Our Action F1:",
        our_action_f1,
        "\n",
        "Our Moveme F1:",
        our_moveme_f1,
    )

    return (all_f1, action_f1, action_f1, moveme_f1)


def results_summary_hbabel(path, sub_nr):
    def _get_top_scores_for_group(res, scores, group="frame", tops=[30, 60, 90]):
        print(f"Results for {group}-level behaviors")
        print_command = ["All F1", "\t\t"] + [f"Top {top} F1" + "\t\t" for top in tops]
        print(*print_command)
        group_action_indices = [
            k for k in res.index if res["Task ID"][k].endswith("_" + group)
        ]
        all_f1 = (
            round(
                res.loc[res["Metric"] == "f1_score", "Private Score"]
                .to_numpy()[group_action_indices]
                .mean(),
                3,
            ),
        )
        all_f1_std = (
            round(
                res.loc[res["Metric"] == "f1_score", scores]
                .to_numpy()[group_action_indices]
                .std(axis=1)
                .mean(),
                3,
            ),
        )
        top_f1 = []
        top_f1_std = []
        for top in tops:
            top_f1.append(
                round(
                    res.loc[res["Metric"] == "f1_score", "Private Score"]
                    .to_numpy()[group_action_indices[-top:]]
                    .mean(),
                    3,
                )
            )
            top_f1_std.append(
                round(
                    res.loc[res["Metric"] == "f1_score", scores]
                    .to_numpy()[group_action_indices[-top:]]
                    .std(axis=1)
                    .mean(),
                    3,
                )
            )

        print_command = [all_f1[0], "\u00B1", all_f1_std[0], "\t"]
        for f1_score, f1_std in list(zip(top_f1, top_f1_std)):
            print_command = print_command + [f1_score, "\u00B1", f1_std, "\t\t"]
        print(*print_command[:-1])

        return all_f1, *top_f1

    res = pd.read_csv(path)
    scores = [f"Score Alpha {x}" for x in [0.1, 0.5, 1.0, 2.0, 5.0]]
    top_frame = [10, 30, 60, 90]
    top_seq = [10, 30, 90]
    f1_scores_frame = _get_top_scores_for_group(
        res, scores, group="frame", tops=top_frame
    )
    f1_scores_seq = _get_top_scores_for_group(res, scores, group="seg", tops=top_seq)

    print("All F1 \t")
    all_f1 = round(
        res.loc[res["Metric"] == "f1_score", "Private Score"].to_numpy().mean(), 3
    )
    all_f1_std = (
        round(
            res.loc[res["Metric"] == "f1_score", scores].to_numpy().std(axis=1).mean(),
            3,
        ),
    )
    print(all_f1, "\u00B1", all_f1_std)
    print("Submission at hierarchy level:", sub_nr)

    return all_f1, f1_scores_frame, f1_scores_seq


if __name__ == "__main__":
    print(f"Starting experiment: {TRY_NAME}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="mabe_mice",
        choices=["mabe_mice", "Shot7M2", "hBABEL"],
    )
    parser.add_argument("--output-dir", default="./results", type=str)
    parser.add_argument("--submission", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument(
        "--training-size",
        "-ts",
        type=str,
        default="75",
        help="Training size in percentage (Experimental for Shot)",
    )
    parser.add_argument(
        "--threshold",
        "-th",
        default=0.5,
        type=float,
        help="Change the prediction threshold (hBABEL experimental)",
    )
    parser.add_argument(
        "--filter",
        "-fi",
        action="store_true",
        help="Filter sequences with no labels (hBABEL experimental)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Set to overwrite existing files"
    )

    args = parser.parse_args()

    if args.task == "mabe_mice":
        task = "mabe_mice"
        dataf = "mouse_triplets"
    elif args.task == "Shot7M2":
        task = args.task
        dataf = "shot_player"
    elif args.task == "hBABEL":
        task = "hBABEL"
        dataf = "hBABEL"

    output_dir = os.path.join(
        args.output_dir,
        args.task
        + "--"
        + args.submission.split(os.sep)[-3]
        + "--"
        + args.submission.split(os.sep)[-2],
    )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if args.threshold == 0.5:
        results_path = os.path.join(
            os.path.dirname(args.submission),
            task + "-round-1-" + "-" + args.submission.split("/")[-1] + ".csv",
        )
    else:
        results_path = os.path.join(
            os.path.dirname(args.submission),
            task
            + "-round-1-"
            + "-"
            + args.submission.split("/")[-1]
            + f"_t{args.threshold}.csv",
        )

    if not os.path.exists(results_path) or args.force:

        # label and submission files
        labels_file = args.labels
        sub_file = args.submission
        output_dir = args.output_dir

        # initialize the evaluator
        evaluator = AIcrowdEvaluator(
            labels_file,
            task_name=task,
            training_size=args.training_size,
            threshold=args.threshold,
            apply_filter=args.filter,
        )

        client_payload = {"submission_file_path": sub_file}
        results = evaluator._evaluate(client_payload)

        print(results)

        # save results to submission dir as well
        results.to_csv(results_path)

        results_path = os.path.join(
            output_dir, task + "-round-1-" + "-" + sub_file.split("/")[-1] + ".csv"
        )
        results.to_csv(results_path, index=False)

        print("\n")

    if True:
        submission_nr = (
            args.submission.split("/")[-1].split("submission")[-1].split(".")[0]
        )

        if args.task == "mabe_mice":
            results_summary_mice(results_path, submission_nr)

        elif args.task == "Shot7M2":
            results_summary_shot7m2(
                results_path, submission_nr, filter_activity_player=True
            )

        elif args.task == "hBABEL":
            results_summary_hbabel(results_path, submission_nr)
