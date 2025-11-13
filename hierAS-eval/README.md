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
    --output_dir [OUTPUT_DIR] \
    --partition_method [random-0.2; mabe_split] \
    --partition_path [only if mabe_split : ./split_files/split_info....json] \
    --dataset [shot7m2; hbabel; mabe_mice; custom]
```

### Custom Datasets

The evaluator supports custom datasets by specifying `--dataset custom`. For custom datasets:

1. **Embeddings format**: Should follow the standard format with `test_submission_*.npy` files containing embeddings
2. **Labels format**: Should be in .npy format compatible with the MABe22 format
3. **Summary output**: Will automatically detect available metrics (F1 score, MSE) and provide:
   - Overall performance statistics
   - Task-level breakdown (if Task ID column exists)
   - Sequence vs Frame level breakdown (if applicable)
   - Standard deviation across multiple evaluation seeds

Example for custom dataset:
```bash
python linear_prober.py \
    --embeddings_path /path/to/your/embeddings/ \
    --labels_path /path/to/your/labels.npy \
    --output_dir results \
    --partition_method random-0.2 \
    --dataset custom
```

### Python version
Python = 3.10 or higher

### Evaluator details

The internal flow of the submissions is described [here](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022#submission).

**Training details** - All models trained use linear models using Scikit-Learn using ridge regression. `Ridge` for regression tasks and `RidgeClassifier` for binary classification tasks. Additionally three seeds are trained for every model where the seed is used to split the dataset 90/10 for training and validation. For classification tasks, the `class_weights` parameter is set to `balanced` for both rounds.

**Scoring** - For binary tasks, predictions are taken via 2/3 vote. For regression tasks, predictions are averaged over all seeds. Once predictions are merged, the score calculated with MSE for regression tasks and F1 score for classification tasks.
