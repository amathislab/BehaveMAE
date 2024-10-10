import json
import os
import subprocess

import pandas as pd
from eval import SingleTaskEvaluator


def get_task_lists():
    TASK_LIST_FILE = os.getenv("TASK_LIST_FILE", "./example_data/task_list.txt")

    TRAIN_CONFIG_FILE = os.getenv("TRAIN_CONFIG_FILE", "training_config.json")

    with open(TASK_LIST_FILE, "r") as f:
        task_list = [line.strip() for line in f.readlines()]

    with open(TRAIN_CONFIG_FILE, "r") as fp:
        training_config = json.load(fp)
        seed_list = training_config["seeds"]

    return task_list, seed_list


def train_all_tasks(task_list, seed_list):
    for task_id in task_list:
        for seed in seed_list:
            cmd = f"python train_single_task.py --seed {seed} --task-id {task_id}"
            cmd_list = cmd.split(" ")
            subprocess.run(cmd_list)
            print(f"Task {task_id} Seed {seed} complete")


def generate_best_runs_report(task_list, seed_list):

    LOG_PATH = os.getenv("TRAINING_LOG_PATH", "./temp")

    best_runs = []

    for task_id in task_list:
        for seed in seed_list:
            report_path = os.path.join(LOG_PATH, f"{task_id}_seed{seed}_runs.csv")
            report = pd.read_csv(report_path)
            best_run_path = report["run_path"][report["val_loss"].idxmin()]
            best_runs.append([task_id, seed, best_run_path])
    best_runs_df = pd.DataFrame(best_runs, columns=["task_id", "seed", "best_run_path"])

    best_runs_report = os.path.join(LOG_PATH, "best_runs.csv")
    best_runs_df.to_csv(best_runs_report, index=False)


def evaluate_single_task(task_id, seed_list, snippet_metadata_path, best_runs_df):

    submission_data_path = os.getenv(
        "SUBMISSION_DATA_PATH", "./example_data/submission_embedding.npy"
    )

    eval = SingleTaskEvaluator(
        task_id=task_id,
        snippet_metadata_path=snippet_metadata_path,
        submission_data_path=submission_data_path,
    )

    task_results = []
    eval.load_data()
    for seed in seed_list:
        best_run_path = best_runs_df[best_runs_df["task_id"] == task_id][
            "best_run_path"
        ]
        result = eval.load_and_evaluate(results_path=best_run_path)
        task_results.append([seed, result])

    return task_results
