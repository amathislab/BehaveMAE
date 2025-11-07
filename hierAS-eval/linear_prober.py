import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LinearProber:
    def __init__(
        self,
        embeddings_path: str,
        labels_path: str,
        output_dir: str,
        seeds: List[int] = [41, 42, 43],
        partition_method: str = "random-0.2",
        partition_file: str = None,
    ):
        """Initialize linear probing setup

        Args:
            embeddings_path: Path to .npy file with frame embeddings
            labels_path: Path to .npy file with frame labels
            output_dir: Where to save results
            seeds: Random seeds for multiple runs
            test_size: Fraction of data to use for testing
        """
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.seeds = seeds
        self.partition_method = partition_method
        self.partition_file = partition_file

        os.makedirs(output_dir, exist_ok=True)

        # Load data
        self.load_data()

    def load_data(self) -> None:
        """Load embeddings and labels"""
        # Load embeddings
        data = np.load(self.embeddings_path, allow_pickle=True).item()
        self.embeddings = data["embeddings"]

        # Load labels
        labels = np.load(self.labels_path, allow_pickle=True).item()
        self.label_array = labels["label_array"]
        self.vocabulary = labels["vocabulary"]
        self.task_types = labels["task_type"]
        if "frame_number_map" in labels.keys():
            self.frame_number_map = labels["frame_number_map"]

    def load_partition(self, partition_file: str) -> Dict:
        if partition_file.endswith(".json"):
            with open(partition_file, "r") as f:
                partition = json.load(f)
        elif partition_file.endswith(".npy"):
            partition = np.load(partition_file, allow_pickle=True).item()
        else:
            raise NotImplementedError("Partition file must be .json or .npy format")
        return partition

    def split_data(self, y: list, seed: int = 0):

        if self.partition_method.startswith("random"):
            test_size = float(self.partition_method.split("-")[1])
            X_train, X_test, y_train, y_test = train_test_split(
                self.embeddings, y, test_size=test_size, random_state=seed
            )

        elif self.partition_method == "mabe_split":
            assert self.partition_file is not None, "Partition file must be provided"
            # Load partition file
            partition = self.load_partition(self.partition_file)
            train_keys = partition["SubmissionTrain"]
            test_keys = partition["publicTest"]

            # Split train
            mask = np.zeros(len(y), dtype=bool)
            for key in train_keys:
                mask[self.frame_number_map[key][0] : self.frame_number_map[key][1]] = (
                    True
                )
            X_train = self.embeddings[mask]
            y_train = y[mask]

            # Split test
            mask = np.zeros(len(y), dtype=bool)
            for key in test_keys:
                mask[self.frame_number_map[key][0] : self.frame_number_map[key][1]] = (
                    True
                )
            X_test = self.embeddings[mask]
            y_test = y[mask]

        return X_train, X_test, y_train, y_test

    def train_eval_single_task(self, task_idx: int, seed: int) -> Tuple[float, float]:
        """Train and evaluate a single task with one seed

        Args:
            task_idx: Index of the task/label to train on
            seed: Random seed

        Returns:
            train_score: Score on training set
            test_score: Score on test set
        """
        # Get labels for this task
        y = self.label_array[task_idx]

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(y, seed)

        # Initialize model based on task type
        if self.task_types[task_idx] == "Discrete":
            model = RidgeClassifier(alpha=1.0, class_weight="balanced")
            score_fn = f1_score
        else:
            model = Ridge(alpha=1.0)
            score_fn = mean_squared_error

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        train_score = score_fn(y_train, model.predict(X_train))
        test_score = score_fn(y_test, model.predict(X_test))

        return train_score, test_score

    def evaluate_all_tasks(self) -> Dict:
        """Evaluate all tasks across all seeds

        Returns:
            results: Dictionary with results for each task and seed
        """
        results = {}
        for task_idx, task_name in tqdm(
            enumerate(self.vocabulary), total=len(self.vocabulary)
        ):
            task_results = {
                "train_scores": [],
                "test_scores": [],
                "task_type": self.task_types[task_idx],
            }

            for seed in self.seeds:
                train_score, test_score = self.train_eval_single_task(task_idx, seed)
                task_results["train_scores"].append(train_score)
                task_results["test_scores"].append(test_score)

            # Calculate mean and std
            task_results["mean_train"] = np.mean(task_results["train_scores"])
            task_results["std_train"] = np.std(task_results["train_scores"])
            task_results["mean_test"] = np.mean(task_results["test_scores"])
            task_results["std_test"] = np.std(task_results["test_scores"])

            results[task_name] = task_results

        # Save result
        file_ext = os.path.basename(self.embeddings_path.split(".")[0])
        output_path = os.path.join(self.output_dir, f"linear_probing_{file_ext}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Linear Probing Evaluation")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to folder with test_submission.npy files",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to .npy file with frame labels",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[41, 42, 43],
        help="Random seeds for multiple runs",
    )
    parser.add_argument(
        "--partition_method",
        type=str,
        default="random-0.2",
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--partition_path",
        type=str,
        required=False,
        help="Path to partition file if partition_method is file",
    )
    return parser.parse_args()


def get_paths(dir_path: str):
    """Get paths to test_submission.npy files in a directory"""

    paths = [f for f in os.listdir(dir_path) if f.startswith("test_submission")]
    if len(paths) == 0:
        raise ValueError(f"No files found in {dir_path} with prefix 'test_submission'")
    return [os.path.join(dir_path, f) for f in paths]


def main(
    embedding_path, labels_path, output_dir, seeds, partition_method, partition_path
):
    prober = LinearProber(
        embeddings_path=embedding_path,
        labels_path=labels_path,
        output_dir=output_dir,
        seeds=seeds,
        partition_method=partition_method,
        partition_file=partition_path,
    )
    # Run evaluation
    results = prober.evaluate_all_tasks()

    # Print results
    for task_name, task_results in results.items():
        print(f"\nTask: {task_name}")
        print(f"Type: {task_results['task_type']}")
        print(
            f"Mean test score: {task_results['mean_test']:.3f} ± {task_results['std_test']:.3f}"
        )

    return results


if __name__ == "__main__":

    # Initialize probing
    args = parse_args()
    embeddings_path = get_paths(args.embeddings_path)

    pool = Pool(processes=max(len(embeddings_path), 1))
    results = pool.starmap(
        main,
        [
            (
                embedding_path,
                args.labels_path,
                args.output_dir,
                args.seeds,
                args.partition_method,
                args.partition_path,
            )
            for embedding_path in embeddings_path
        ],
    )
    pool.close()

    print(results)
