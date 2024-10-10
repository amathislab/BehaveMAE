## Submission Format

Embeddings need to be provided for the entire test in one numpy file.

The maximum embedding size allowed is 128 for MABe22 mice, and 64 for both Shot7M2 and hBABEL.

NaN values are not allowed in the embeddings.

## Training pipeline for tasks

Each task is trained with the provided embeddings independently.

### Base neural net trained per task

- Keras 2 layer neural network
- Fixed number of epochs per task, save model with best validation loss
- Activation and loss Function
    - Sigmoid Binary Crossentropy for binary labels
    - tanh with output transform for continuous labels, mse loss


### Hyperparameter search

A grid search of 4 values of the following parameters

- Learning rate - [0.1, 0.03, 0.01, 0.001]
- Hidden layer units - [32, 100, 200, 512]

Total 16 runs per task, the model with lowest validation loss is retained. 

### Multiple seeds

3 seeds are used for splitting the data (90-10 train/validation). For each seed, the full hyperparameter search is performed. Scores of 3 models with best validation loss are averaged.

A total of 16*3 = 48 training runs are done per task.
