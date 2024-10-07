import pickle

import numpy as np
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(
        self,
        submission_data_path,
        labels_path,
        frame_number_map_file,
        split_info,
        dataset,
        apply_filter=False,
    ):

        self.dataset = dataset

        self.submission_data_path = submission_data_path
        self.labels_path = labels_path
        self.frame_number_map_file = frame_number_map_file
        self.split_info = split_info

        self.split_clip_names = None
        self.train_snippets, self.test_snippets = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None

        self.apply_filter = apply_filter
        self.load_embeddings_and_labels()

    def load_embeddings_and_labels(self):

        if "shot" in self.dataset.lower():

            if self.submission_data_path.endswith(".pickle"):
                with open(self.submission_data_path, "rb") as f:
                    sub_dict = pickle.load(f)
                    self.sub_emb = sub_dict["embeddings"]
            elif self.submission_data_path.endswith(".npy"):
                try:
                    sub_dict = np.load(
                        self.submission_data_path, allow_pickle=True
                    ).item()
                    self.sub_emb = sub_dict["embeddings"]
                except:
                    raise ValueError("embeddings format of SHOT is not supported..")
            elif self.submission_data_path.endswith(".dat"):
                self.sub_emb = np.memmap(
                    self.submission_data_path,
                    mode="r",
                    dtype=np.float32,
                    shape=(2176000, 96 * 6),
                )

            if self.labels_path.endswith(".pickle"):
                with open(self.labels_path, "rb") as f:
                    self.labels = pickle.load(f)
            else:
                self.labels = np.load(self.labels_path, allow_pickle=True).item()
            self.frame_number_map = self.labels["frame_number_map"]

        else:
            # MABe dataset
            sub_dict = np.load(self.submission_data_path, allow_pickle=True).item()
            self.frame_number_map = np.load(
                self.frame_number_map_file, allow_pickle=True
            ).item()
            self.sub_emb = sub_dict["embeddings"]
            self.labels = np.load(self.labels_path, allow_pickle=True).item()

            if self.apply_filter:
                self.filter_empty_episodes(
                    self.frame_number_map, self.sub_emb, self.labels
                )

    def filter_empty_episodes(self, frame_number_map, embeddings, labels):
        start_count = 0
        new_label_array = []
        new_frame_number_map = {}
        new_embeddings = []
        for key, (start, stop) in frame_number_map.items():
            clip_duration = stop - start
            label_array = labels["label_array"][:, start:stop]
            # Filter empty sequences
            if np.sum(label_array[1:62]):
                # Filter sequences with few labeled frames with respect to the clip length
                if np.sum(label_array[62:]) > 0.2 * clip_duration:
                    new_label_array.append(label_array)
                    new_embeddings.append(embeddings[start:stop])
                    new_frame_number_map[key] = (
                        start_count,
                        start_count + clip_duration,
                    )
                    start_count += clip_duration

        new_label_array = np.concatenate(new_label_array, axis=1)
        new_embeddings = np.concatenate(new_embeddings, axis=0)
        self.labels["label_array"] = new_label_array
        self.sub_emb = new_embeddings
        self.frame_number_map = new_frame_number_map

    def split_and_load_data(self, seed, split_keys, test_size=0.1):
        self.seed = seed
        self.test_size = test_size
        self.split_keys = split_keys

        self.train_snippets, self.test_snippets = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None

        self._select_by_split_keys()
        self._split_snippet_names()
        self.X_train_full, self.X_val_full = self._split_embeddings()

    def load_labels(self, task_id):
        self.task_id = task_id
        self.y_train, self.y_val = self._load_labels()
        self.remove_nan_indexes()

    def _select_by_split_keys(self):
        self.split_clip_names = []
        for ds in self.split_keys:
            if self.apply_filter:
                filtered_keys = [
                    key for key in self.split_info[ds] if key in self.frame_number_map
                ]
                self.split_clip_names.extend(filtered_keys)
            else:
                self.split_clip_names.extend(self.split_info[ds])

    def _split_snippet_names(self):
        if self.test_size > 0:
            self.train_snippets, self.test_snippets = train_test_split(
                self.split_clip_names, test_size=self.test_size, random_state=self.seed
            )
        else:
            self.train_snippets, self.test_snippets = self.split_clip_names, []

    def remove_nan_indexes(self):
        def rem_nan_idx(X, y):
            y_notnan_idx = ~np.isnan(y)
            X_new = X[y_notnan_idx]
            y_new = y[y_notnan_idx]
            return X_new, y_new

        # y_notnan_idx = ~np.isnan(self.y_train)
        # if np.sum(y_notnan_idx) == 0:
        #     import pdb; pdb.set_trace()
        #     print(self.task_idx, self.task_id)
        #     return

        self.X_train, self.y_train = rem_nan_idx(self.X_train_full, self.y_train)
        if self.test_size > 0:
            self.X_val, self.y_val = rem_nan_idx(self.X_val_full, self.y_val)

    def _get_index(self, snippets, length):
        index = np.zeros(length, bool)
        for sk in snippets:
            start, end = self.frame_number_map[sk]
            index[start:end] = True

        return index

    def _split_embeddings(self):
        self.train_index = self._get_index(self.train_snippets, self.sub_emb.shape[0])
        self.val_index = self._get_index(self.test_snippets, self.sub_emb.shape[0])
        X_train = self.sub_emb[self.train_index]
        X_val = self.sub_emb[self.val_index]
        return X_train, X_val

    def _set_task_info(self):
        self.task_idx = self.labels["vocabulary"].index(self.task_id)
        self.task_type = self.labels["task_type"][self.task_idx]

    def _load_labels(self):

        self._set_task_info()
        y_train = self.labels["label_array"][self.task_idx, self.train_index]
        y_val = self.labels["label_array"][self.task_idx, self.val_index]

        return y_train, y_val
