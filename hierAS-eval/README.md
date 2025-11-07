# Evaluator code for Hierarchical Action Segmentation (hierAS) benchmarks

Code for the evaluation of the embeddings on the MABe22, Shot7M2 and hBABEL datasets.

## Data checklist
In order to run the evaluator, you will need
1. A label file in the MABe22 format (.npy) which contains the behavior ground truth data.
2. A submission file (.npy) which contains the embeddings subject to evaluation.
3. (Optional) a path where to save the results.

## Evaluation

The evaluator supports evaluation for hBABEL, Shot7M2 and MABe22. To start the evaluator, you can run the following
```
python linear_prober \
    --embeddings_path [PATH_TO_EMBEDDINGS] \
    --labels_path [PATH_TO_LABELS] \
    --output-dir [OUTPUT_DIR] \
    --partition_method [random-0.2; mabe_split]
    --partition_path [./split_files/split_info....json]
```
You can download the split file from the MABe22 when downloading the dataset.

### Python version
Python = 3.10 or higher

### Evaluator details

The internal flow of the submissions is described [here](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022#submission).

**Training details** - All models trained use linear models using Scikit-Learn using ridge regression. `Ridge` for regression tasks and `RidgeClassifier` for binary classification tasks. Additionally three seeds are trained for every model where the seed is used to split the dataset 90/10 for training and validation. For classification tasks, the `class_weights` parameter is set to `balanced` for both rounds.

**Scoring** - For binary tasks, predictions are taken via 2/3 vote. For regression tasks, predictions are averaged over all seeds. Once predictions are merged, the score calculated with MSE for regression tasks and F1 score for classification tasks.
