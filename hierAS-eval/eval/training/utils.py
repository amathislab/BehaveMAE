import numpy as np
import pandas as pd


def get_seeded_numpy_state(seed):
    return np.random.RandomState(seed)


class TaskInfo:
    def __init__(self, task_id, snippet_metadata_path):
        self.task_id = task_id
        self.snippet_metadata_df = pd.read_csv(snippet_metadata_path)
        task_match_idx = self.snippet_metadata_df["task_id"] == self.task_id
        self.task_snippets_df = self.snippet_metadata_df[task_match_idx]
        self.label_type = self.task_snippets_df["label_type"].values[0]
