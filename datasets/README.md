# Dataset Preparation

To prepare the datasets required for our hierarchical action segmentation benchmarks, follow the steps below. Each dataset should be placed in its corresponding directory under a main `data` directory.

## Create Directory Structure

First, create the necessary directories:

```bash
mkdir -p data/Shot7M2
mkdir -p data/hBABEL
mkdir -p data/MABe22
```

## Shot7M2

Download the Shot7M2 dataset from [HuggingFace](https://huggingface.co/datasets/amathislab/SHOT7M2) and place it in the `data/Shot7M2` directory.

### Steps:

1. **Ensure Git LFS is installed**:
   ```bash
   git lfs install
   ```
2. **Clone the dataset**:
    ```bash
    git clone https://huggingface.co/datasets/amathislab/SHOT7M2 data/Shot7M2
    ```

## hBABEL Action Segmentation Benchmark

The hBABEL dataset is an extension of the [BABEL dataset](https://babel.is.tue.mpg.de/index.html) developed for hierarchical action segmentation. Please cite both the original BABEL paper and our ECCV paper when using hBABEL.

### Steps:

### A. Downloading Data

1. Follow the instructions on the [TEACH GitHub repository](https://github.com/athn-nik/teach/tree/main?tab=readme-ov-file#data) to download and process the AMASS and BABEL datasets.
   
2. You should end up with the same data folders, as described in the TEACH repository.

### B. Data Conversion to hBABEL

1. **Perform Procrustes alignment on the pose data**:
   ```bash
   python -m datasets.hBABEL.prepare_hbabel_data
   ```
2. **Generate the hBABEL dataset**:
   ```bash
   python datasets/hBABEL/prepare_hbabel_labels.py
   ```

## MABe22

The [MABe 2022](https://sites.google.com/view/computational-behavior/our-datasets/mabe2022-dataset) dataset is publicly available. Follow the steps below to download the mouse triplet data:

```bash
wget "https://data.caltech.edu/records/8kdn3-95j37/files/mouse_triplet_train.npy" -O "data/MABe22/mouse_triplet_train.npy"
wget "https://data.caltech.edu/records/8kdn3-95j37/files/mouse_triplet_test.npy" -O "data/MABe22/mouse_triplet_test.npy"
wget "https://data.caltech.edu/records/8kdn3-95j37/files/mouse_triplets_test_labels.npy" -O "data/MABe22/mouse_triplets_test_labels.npy"
```
