# Evaluator code for Hierarchical Action Segmentation (hierAS) benchmarks

Based on the evaluator code from the [MABe 2022 Challenge](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022). Thanks to the authors for their great work!

## Data checklist
In order to run the evaluator, you will need
1. A label file in the MABe22 format (.npy) which contains the behavior ground truth data.
2. A submission file (.npy) which contains the embeddings subject to evaluation.
3. (Optional) a path where to save the results.

## Evaluation

The evaluator supports evaluation for hBABEL, Shot7M2 and MABe22. To start the evaluator, you can run the following
```
python evaluator \
    --task [choose between: mabe_mice, hBABEL, Shot7M2] \
    --submission [PATH_TO_SUBMISSION_FILE] \
    --labels [PATH_TO_LABELS] \
    --output-dir [OUTPUT_DIR]
```

### Python version and packages

Originally used with Python 3.9 - But should work with any python version above 3.6

Originally used packages:
```
numpy==1.24.3
scikit-learn==1.2.2
pandas==2.0.1
tqdm==4.65.0
```

### Evaluator details

The internal flow of the submissions is described [here](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022#submission).

**Training details** - All models trained use linear models using Scikit-Learn using ridge regression. `Ridge` for regression tasks and `RidgeClassifier` for binary classification tasks. Additionally three seeds are trained for every model where the seed is used to split the dataset 90/10 for training and validation. For classification tasks, the `class_weights` parameter is set to `balanced` for both rounds.

**Scoring** - For binary tasks, predictions are taken via 2/3 vote. For regression tasks, predictions are averaged over all seeds. Once predictions are merged, the score calculated with MSE for regression tasks and F1 score for classification tasks.

## Modifications from the original evaluator
We here detail the modifications from the original [mabe22-eval](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022):

* **Input data loading** : hierAS-eval supports the loading of hBABEL and Shot7M2 meta data files.
* **Output file generation** : hierAS-eval generates result files adapted for the hBABEL and Shot7M2 datasets.

No change has been performed on the training and evaluation protocols.
