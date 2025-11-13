import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="shot7m2",
        help="Dataset type for generating appropriate summary statistics. "
             "Use 'shot7m2', 'hbabel', 'mabe_mice' for built-in summaries, "
             "or 'custom' (or any other name) for generic summary.",
    )
    return parser.parse_args()


def get_paths(dir_path: str):
    """Get paths to test_submission.npy files in a directory"""

    paths = [f for f in os.listdir(dir_path) if f.startswith("test_submission")]
    if len(paths) == 0:
        raise ValueError(f"No files found in {dir_path} with prefix 'test_submission'")
    return [os.path.join(dir_path, f) for f in paths]


def extract_hierarchy_levels(embeddings_paths):
    """Extract hierarchy levels from submission filenames"""
    import re
    hierarchy_levels = []
    for path in embeddings_paths:
        filename = os.path.basename(path)
        # Extract number from test_submission_X.npy
        match = re.search(r'test_submission_(\d+)\.npy', filename)
        if match:
            hierarchy_levels.append(int(match.group(1)))
        else:
            # Fallback to index if no number found
            hierarchy_levels.append(len(hierarchy_levels))
    return hierarchy_levels


def results_summary_generic(results_dir, embeddings_paths=None):
    """Generate generic results summary for custom datasets"""
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        print("No JSON result files found in", results_dir)
        return
    
    print(f"Found {len(json_files)} result file(s)")
    
    # Extract hierarchy levels if embeddings paths are provided
    hierarchy_levels = []
    if embeddings_paths:
        hierarchy_levels = extract_hierarchy_levels(embeddings_paths)
    
    for i, json_file in enumerate(json_files):
        # Use extracted hierarchy level if available, otherwise use index
        level = hierarchy_levels[i] if i < len(hierarchy_levels) else i
        print(f"\nSubmission at hierarchy level: {level}")
        print(f"Processing: {json_file}")
        
        json_path = os.path.join(results_dir, json_file)
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        print(f"Number of tasks: {len(results)}")
        
        # Collect F1 scores (test scores)
        all_test_scores = []
        all_test_stds = []
        task_results = []
        
        for task_name, task_data in results.items():
            test_score = task_data.get('mean_test', 0)
            test_std = task_data.get('std_test', 0)
            task_type = task_data.get('task_type', 'Unknown')
            
            all_test_scores.append(test_score)
            all_test_stds.append(test_std)
            task_results.append((task_name, test_score, test_std, task_type))
        
        # Overall summary
        overall_score = round(np.mean(all_test_scores), 3)
        overall_std = round(np.mean(all_test_stds), 3)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Mean Score: {overall_score} ± {overall_std}")
        
        # Group by task type prefix for hierarchical analysis
        task_groups = {}
        for task_name, score, std, task_type in task_results:
            prefix = task_name.split('_')[0] if '_' in task_name else 'other'
            if prefix not in task_groups:
                task_groups[prefix] = []
            task_groups[prefix].append((task_name, score, std, task_type))
        
        # Show breakdown by task group
        print(f"\n� Task Group Breakdown:")
        for group_name, tasks in task_groups.items():
            group_scores = [score for _, score, _, _ in tasks]
            group_mean = round(np.mean(group_scores), 3)
            group_std = round(np.std(group_scores), 3) if len(group_scores) > 1 else 0
            print(f"   {group_name.title()}: {group_mean} ± {group_std} ({len(tasks)} tasks)")
        
        # Show individual task results
        print(f"\n📝 Individual Task Results:")
        for task_name, score, std, task_type in sorted(task_results):
            print(f"     {task_name}: {score} ± {std} ({task_type})")
        
        print()
        print("_" * 50)


def results_summary_mice(results_dir, embeddings_paths=None):
    """Generate MABe22 mice results summary"""
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        print("No JSON result files found in", results_dir)
        return
    
    # Extract hierarchy levels if embeddings paths are provided
    hierarchy_levels = []
    if embeddings_paths:
        hierarchy_levels = extract_hierarchy_levels(embeddings_paths)
    
    for i, json_file in enumerate(json_files):
        # Use extracted hierarchy level if available, otherwise use index
        level = hierarchy_levels[i] if i < len(hierarchy_levels) else i
        print(f"\nSubmission at hierarchy level: {level}")
        
        json_path = os.path.join(results_dir, json_file)
        with open(json_path, 'r') as f:
            results = json.load(f)

        print("All MSE", "\t\t", "All F1", "\t\t", "Frame F1", "\t\t", "Sequence F1")
        
        # Separate tasks by type - assuming task names indicate whether they are MSE or F1 tasks
        # and whether they are frame-level or sequence-level
        f1_tasks = []
        mse_tasks = []
        frame_tasks = []
        seq_tasks = []
        
        for task_name, task_data in results.items():
            task_type = task_data.get('task_type', 'Discrete')
            test_score = task_data.get('mean_test', 0)
            test_std = task_data.get('std_test', 0)
            
            if task_type == 'Discrete':  # F1 score tasks
                f1_tasks.append((test_score, test_std))
                # Assume sequence-level tasks have certain keywords in their names
                if any(keyword in task_name.lower() for keyword in ['sequence', 'seq', 'episode']):
                    seq_tasks.append((test_score, test_std))
                else:
                    frame_tasks.append((test_score, test_std))
            else:  # MSE tasks (continuous)
                mse_tasks.append((test_score, test_std))
        
        # Calculate MSE statistics
        if mse_tasks:
            all_mse = round(np.mean([score for score, _ in mse_tasks]), 4)
            all_mse_std = round(np.mean([std for _, std in mse_tasks]), 4)
        else:
            all_mse, all_mse_std = 0.0, 0.0
            
        # Calculate F1 statistics
        if f1_tasks:
            all_f1 = round(np.mean([score for score, _ in f1_tasks]), 3)
            all_f1_std = round(np.mean([std for _, std in f1_tasks]), 3)
        else:
            all_f1, all_f1_std = 0.0, 0.0
            
        # Frame-level F1
        if frame_tasks:
            frame_f1 = round(np.mean([score for score, _ in frame_tasks]), 3)
            frame_f1_std = round(np.mean([std for _, std in frame_tasks]), 3)
        else:
            frame_f1, frame_f1_std = 0.0, 0.0
            
        # Sequence-level F1
        if seq_tasks:
            seq_f1 = round(np.mean([score for score, _ in seq_tasks]), 3)
            seq_f1_std = round(np.mean([std for _, std in seq_tasks]), 3)
        else:
            seq_f1, seq_f1_std = 0.0, 0.0

        print(
            all_mse,
            "\u00B1",
            all_mse_std,
            "\t",
            all_f1,
            "\u00B1",
            all_f1_std,
            "\t\t",
            frame_f1,
            "\u00B1",
            frame_f1_std,
            "\t\t",
            seq_f1,
            "\u00B1",
            seq_f1_std,
        )
        print()


def results_summary_hbabel(results_dir, embeddings_paths=None):
    """Generate hBABEL results summary"""
    def _get_top_scores_for_group(task_data, group="frame", tops=[30, 60, 90]):
        print(f"Results for {group}-level behaviors")
        print_command = ["All F1", "\t\t"] + [f"Top {top} F1" + "\t\t" for top in tops]
        print(*print_command)
        
        # Filter tasks that end with the group suffix
        group_tasks = [(name, data) for name, data in task_data.items() 
                       if name.endswith("_" + group)]
        
        if not group_tasks:
            print("No tasks found for group:", group)
            return (0.0,) + tuple([0.0] * len(tops))
        
        # Sort by task name to ensure consistent ordering
        group_tasks.sort(key=lambda x: x[0])
        
        # Get scores and calculate statistics
        group_scores = [data['mean_test'] for _, data in group_tasks]
        group_stds = [data['std_test'] for _, data in group_tasks]
        
        all_f1 = round(np.mean(group_scores), 3)
        all_f1_std = round(np.mean(group_stds), 3)
        
        top_f1 = []
        top_f1_std = []
        for top in tops:
            if len(group_scores) >= top:
                # Take the last 'top' scores (assuming sorted by difficulty)
                top_scores = group_scores[-top:]
                top_stds_vals = group_stds[-top:]
            else:
                # If we don't have enough tasks, use all available
                top_scores = group_scores
                top_stds_vals = group_stds
                
            top_f1.append(round(np.mean(top_scores), 3))
            top_f1_std.append(round(np.mean(top_stds_vals), 3))

        print_command = [all_f1, "\u00B1", all_f1_std, "\t"]
        for f1_score, f1_std in list(zip(top_f1, top_f1_std)):
            print_command = print_command + [f1_score, "\u00B1", f1_std, "\t\t"]
        print(*print_command[:-1])

        return (all_f1,) + tuple(top_f1)

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        print("No JSON result files found in", results_dir)
        return
    
    # Extract hierarchy levels if embeddings paths are provided
    hierarchy_levels = []
    if embeddings_paths:
        hierarchy_levels = extract_hierarchy_levels(embeddings_paths)
    
    for i, json_file in enumerate(json_files):
        # Use extracted hierarchy level if available, otherwise use index
        level = hierarchy_levels[i] if i < len(hierarchy_levels) else i
        print(f"\nSubmission at hierarchy level: {level}")
        
        json_path = os.path.join(results_dir, json_file)
        with open(json_path, 'r') as f:
            results = json.load(f)
            
        top_frame = [10, 30, 60, 90]
        top_seq = [10, 30, 90]
        f1_scores_frame = _get_top_scores_for_group(
            results, group="frame", tops=top_frame
        )
        f1_scores_seq = _get_top_scores_for_group(results, group="seg", tops=top_seq)

        print("All F1 \t")
        all_scores = [task_data['mean_test'] for task_data in results.values()]
        all_stds = [task_data['std_test'] for task_data in results.values()]
        all_f1 = round(np.mean(all_scores), 3)
        all_f1_std = round(np.mean(all_stds), 3)
        print(all_f1, "\u00B1", all_f1_std)
        print()


def results_summary_shot7m2(results_dir, embeddings_paths=None, filter_activity_player=False):
    """Generate Shot7M2 results summary similar to the original evaluator"""
    
    # Find all JSON result files
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        print("No JSON result files found in", results_dir)
        return
    
    # Extract hierarchy levels if embeddings paths are provided
    hierarchy_levels = []
    if embeddings_paths:
        hierarchy_levels = extract_hierarchy_levels(embeddings_paths)
    
    # Process each submission
    for i, json_file in enumerate(json_files):
        # Use extracted hierarchy level if available, otherwise use index
        level = hierarchy_levels[i] if i < len(hierarchy_levels) else i
        print(f"\nSubmission at hierarchy level: {level}")
        
        json_path = os.path.join(results_dir, json_file)
        with open(json_path, 'r') as f:
            results = json.load(f)

        def compute_our_movemes_mean(scores):
            scores_arr = np.array(scores)
            scores1 = scores_arr[:6]
            scores2 = scores_arr[6:].reshape(-1, 2).mean(axis=1)
            scores_ = np.concatenate([scores1, scores2])
            return scores_.mean()

        def compute_our_actions_mean(scores):
            scores_arr = np.array(scores)
            scores1 = scores_arr[:6]
            scores2 = scores_arr[8:10]
            scores3 = scores_arr[6:8].reshape(-1, 2).mean(axis=1)
            scores4 = scores_arr[10:].reshape(-1, 2).mean(axis=1)
            scores_ = np.concatenate([scores1, scores2, scores3, scores4])
            return scores_.mean()

        # Collect all F1 scores
        all_scores = [task_data['mean_test'] for task_data in results.values()]
        all_f1 = round(np.mean(all_scores), 3)
        all_f1_std = round(np.mean([task_data['std_test'] for task_data in results.values()]), 3)

        print("All F1")
        print(f"{all_f1} ± ({all_f1_std},)")

        # Hierarchical breakdown
        print("Activity F1", "\t\t\t", "Action F1", "\t\t\t", "Moveme F1")
        
        # Activity tasks
        activity_tasks = [name for name in results.keys() if name.startswith("activity")]
        if filter_activity_player:
            activity_tasks = [name for name in results.keys() if name.startswith("activity_Episode")]
        
        activity_scores = [results[name]['mean_test'] for name in activity_tasks] if activity_tasks else []
        activity_f1 = round(np.mean(activity_scores), 3) if activity_scores else 0
        activity_f1_std = round(np.mean([results[name]['std_test'] for name in activity_tasks]), 3) if activity_tasks else 0

        # Action tasks
        action_tasks = [name for name in results.keys() if name.startswith("action")]
        action_scores = [results[name]['mean_test'] for name in action_tasks] if action_tasks else []
        action_f1 = round(np.mean(action_scores), 3) if action_scores else 0
        action_f1_std = round(np.mean([results[name]['std_test'] for name in action_tasks]), 3) if action_tasks else 0

        # Moveme tasks
        moveme_tasks = [name for name in results.keys() if name.startswith("moveme")]
        moveme_scores = [results[name]['mean_test'] for name in moveme_tasks] if moveme_tasks else []
        moveme_f1 = round(np.mean(moveme_scores), 3) if moveme_scores else 0
        moveme_f1_std = round(np.mean([results[name]['std_test'] for name in moveme_tasks]), 3) if moveme_tasks else 0

        print(
            activity_f1, "\u00B1", activity_f1_std, "\t\t",
            action_f1, "\u00B1", action_f1_std, "\t\t\t",
            moveme_f1, "\u00B1", moveme_f1_std
        )

        # Compute "Our" scores
        if action_scores:
            our_action_f1 = compute_our_actions_mean(action_scores)
        else:
            our_action_f1 = 0
            
        if moveme_scores:
            our_moveme_f1 = compute_our_movemes_mean(moveme_scores)
        else:
            our_moveme_f1 = 0
            
        our_all_f1 = np.mean([activity_f1, our_action_f1, our_moveme_f1])

        print()
        print(f"Our All F1: {our_all_f1}")
        print(f"Our Activity F1: {activity_f1}")
        print(f"Our Action F1: {our_action_f1}")
        print(f"Our Moveme F1: {our_moveme_f1}")
        print()


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
    
    # Generate dataset-specific summary
    print("\n" + "="*60)
    print(f"RESULTS SUMMARY - {args.dataset.upper()}")
    print("="*60)
    
    dataset_lower = args.dataset.lower()
    if dataset_lower == "shot7m2":
        results_summary_shot7m2(args.output_dir, embeddings_path, filter_activity_player=True)
    elif dataset_lower == "hbabel":
        results_summary_hbabel(args.output_dir, embeddings_path)
    elif dataset_lower == "mabe_mice":
        results_summary_mice(args.output_dir, embeddings_path)
    else:
        # Use generic summary for custom datasets or any unrecognized dataset names
        print(f"Using generic summary for dataset: {args.dataset}")
        results_summary_generic(args.output_dir, embeddings_path)
