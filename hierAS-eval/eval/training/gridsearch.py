import itertools
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from trainer import SingleTaskTrainer


class HParamGridSearcher:
    def __init__(self, trainer, hparams_dict, log_path):
        self.trainer = trainer
        self.hparams_dict = hparams_dict
        self.log_path = log_path
        self.all_run_results = {"run_prefix": [], "val_loss": [], "run_path": []}

    def run_hparams_grid(self):

        hparams_list = list(itertools.product(*self.hparams_dict.values()))

        run_description = f"{self.trainer.taskinfo.task_id}_seed{self.trainer.seed}"
        for idx, hparams in enumerate(tqdm(hparams_list, desc=run_description)):

            training_params = {k: v for k, v in zip(self.hparams_dict, hparams)}

            run_prefix = f"run{idx}"

            self.trainer.setup_logging(log_path=self.log_path, train_prefix=run_prefix)
            self.trainer.setup_neural_net(training_params=training_params)
            self.trainer.train()

            val_loss = self.trainer.evaluate()

            self.all_run_results["run_prefix"].append(run_prefix)
            self.all_run_results["val_loss"].append(val_loss)
            self.all_run_results["run_path"].append(self.trainer.log_path)

    def get_best_model_runid(self):
        assert (
            len(self.all_run_results["run_prefix"]) > 0
        ), "Results list empty, run training first"
        val_loss = self.all_run_results["val_loss"]
        run_prefix = self.all_run_results["run_prefix"]
        return run_prefix[val_loss.index(min(val_loss))]


class SingleSeedTrainer:
    def __init__(
        self,
        seed,
        task_id,
        snippet_metadata_path,
        submission_data_path,
        log_path,
        hparams_dict,
    ):
        self.seed = seed
        self.trainer = SingleTaskTrainer(
            task_id=task_id,
            snippet_metadata_path=snippet_metadata_path,
            submission_data_path=submission_data_path,
            seed=seed,
        )

        self.hparam_searcher = HParamGridSearcher(
            trainer=self.trainer, hparams_dict=hparams_dict, log_path=log_path
        )

        report_name = f"{task_id}_seed{seed}_runs.csv"
        self.report_path = os.path.join(log_path, report_name)

    def train(self):
        self.trainer.load_data()
        self.hparam_searcher.run_hparams_grid()

    def save_report(self):
        report_df = pd.DataFrame(self.hparam_searcher.all_run_results)
        report_df.to_csv(self.report_path, index=False)
        return report_df
