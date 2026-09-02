# Rice Grain Classification with CNN

This project classifies images of rice grains into five varieties using a convolutional neural network built with TensorFlow/Keras. The notebook covers loading and exploring the image dataset, building a CNN from scratch, training it, and evaluating how well it separates the five rice types.

## Overview

Different rice varieties (Arborio, Basmati, Ipsala, Jasmine, Karacadag) look fairly similar at a glance but have subtle differences in shape, size, and texture that a human might struggle to describe but a CNN can pick up on from pixel data. The goal here was to train an image classifier that takes a photo of a single rice grain and predicts which of the five varieties it belongs to.

The approach is a straightforward supervised image classification pipeline: load the images, split them into train/validation/test sets, normalize pixel values, train a small custom CNN, and check the results with standard classification metrics.

## Dataset

The notebook uses the **Rice Image Dataset** (`muratkokludataset/rice-image-dataset`), loaded from a local path pointing at a Kaggle dataset directory: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

- **Total images:** 75,000
- **Classes (5):** Arborio, Basmati, Ipsala, Jasmine, Karacadag
- **Class balance:** perfectly balanced — 15,000 images per class
- **Format:** the images live in one folder per class, and the notebook also builds a Pandas DataFrame of `image_path` / `label` pairs for the class distribution check

Since every class has exactly 15,000 images, there's no class-imbalance problem to worry about here.

## Workflow

1. Load and inspect the image folders
2. Visualize sample images from each class
3. Build an `image_path` / `label` DataFrame and check class distribution
4. Split the data into train / validation / test
5. Load images as TensorFlow datasets and normalize pixel values
6. Visualize normalized images to confirm preprocessing worked
7. Build a CNN from scratch
8. Compile the model and set up training callbacks
9. Train the model
10. Plot accuracy/loss curves
11. Evaluate on the test set (classification report + confusion matrix)
12. Run a prediction on a random sample image

## Exploratory Data Analysis

The EDA here is intentionally light since this is an image dataset rather than tabular data:

- A grid of sample images (4 per class) is plotted to visually check what each rice variety looks like.
- A bar chart of class counts confirms the dataset is evenly split across all five classes (15,000 each).

There's no missing-value or outlier analysis, which makes sense for a folder of labeled images rather than a CSV of numeric features.

## Data Preprocessing

- **Train/val/test split:** the notebook actually creates two separate splits. First, `train_test_split` from scikit-learn splits the `image_path`/`label` DataFrame into `train_df` (70%), and a `temp_df` that's further split 50/50 into `val_df` and `test_df` (15% each). This DataFrame-based split is built and printed but isn't the one that's actually fed into the model.
- **Actual data pipeline:** training instead uses Keras' `image_dataset_from_directory` directly on the full image folder, with `validation_split=0.2` giving 60,000 training images and 15,000 held out. That held-out 15,000 is then split again in half (using `tf.data` batch slicing) into a validation set and a test set.
- **Resizing:** all images are resized to 128×128 as they're loaded.
- **Batching:** batch size of 32.
- **Normalization:** pixel values are scaled from `[0, 255]` to `[0, 1]` by dividing by 255. This helps the network train faster and more stably than feeding in raw pixel intensities.
- **Performance:** `.cache()` and `.prefetch(AUTOTUNE)` are applied to the datasets to speed up the data loading pipeline during training.
- **Labels:** one-hot encoded (`label_mode="categorical"`) to match the softmax output and categorical cross-entropy loss.

Worth noting: because the DataFrame split and the `image_dataset_from_directory` split aren't the same split, the `train_df`/`val_df`/`test_df` variables from step 6 end up unused for training — the model is trained on the dataset objects from `image_dataset_from_directory` instead.

## Model

A single custom CNN was built from scratch using `tf.keras.models.Sequential` — no pretrained/transfer-learning backbone is used.

**Architecture:**
- 3 convolutional blocks, each with `Conv2D → BatchNormalization → MaxPooling2D`
  - Block 1: 32 filters, 3×3 kernel
  - Block 2: 64 filters, 3×3 kernel
  - Block 3: 64 filters, 3×3 kernel
- `Flatten`
- `Dense(64, activation="relu")`
- `Dropout(0.5)`
- `Dense(5, activation="softmax")` output layer

Batch normalization after each conv layer helps stabilize training, and dropout before the final layer helps reduce overfitting given the fully-connected layer's parameter count. The final model has **1,105,925 total parameters** (1,105,605 trainable, 320 non-trainable from the BatchNorm layers).

## Training

- **Optimizer:** Adam
- **Loss:** categorical cross-entropy
- **Metric:** accuracy
- **Epochs:** configured for up to 5
- **Batch size:** 32
- **Callbacks:**
  - `ReduceLROnPlateau` — halves the learning rate if `val_loss` doesn't improve for 2 epochs
  - `EarlyStopping` — stops training if `val_loss` doesn't improve for 5 epochs, restoring the best weights
  - `ModelCheckpoint` — saves the model with the best `val_accuracy`
- **Hardware:** the training logs show the notebook was run on a GPU (Kaggle environment, 2x Tesla T4)

Training ran the full 5 configured epochs (early stopping never triggered since patience was 5). The learning rate was reduced once, after epoch 3.

## Evaluation

The model was evaluated on the held-out test set using:

- **Accuracy** — overall fraction of correctly classified grains
- **Precision / Recall / F1-score** (per class, via `classification_report`) — useful here even though classes are balanced, to check if any single rice variety is harder to distinguish than the others
- **Confusion Matrix** — to see which classes (if any) get confused with each other

## Results

Training accuracy/loss by epoch:

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|----------------|------------|---------------|----------|
| 1     | 0.8766         | 0.2955     | 0.9472        | 0.2049   |
| 2     | 0.9409         | 0.1604     | 0.9444        | 0.2279   |
| 3     | 0.9532         | 0.1316     | 0.9316        | 0.2338   |
| 4     | 0.9750         | 0.0798     | 0.9866        | 0.0485   |
| 5     | 0.9806         | 0.0664     | 0.9955        | 0.0178   |

**Test set performance:** 0.9956 accuracy, 0.0177 loss (7,488 test images).

Per-class results from the classification report (test set):

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Arborio   | 1.00      | 0.99   | 0.99     | 1480    |
| Basmati   | 0.99      | 1.00   | 1.00     | 1545    |
| Ipsala    | 1.00      | 1.00   | 1.00     | 1430    |
| Jasmine   | 0.99      | 0.99   | 0.99     | 1521    |
| Karacadag | 0.99      | 1.00   | 1.00     | 1512    |

All five classes end up with essentially the same near-perfect precision, recall, and F1-score (0.99–1.00), so there isn't a "weak" class here — the model separates all five rice varieties equally well.

## Visual Results

The notebook produces a few plots worth mentioning if you're browsing the repo:

- **Sample grids** of raw and normalized images per class, useful for sanity-checking that the images loaded and preprocessed correctly.
- **Class distribution bar chart**, confirming the 15,000-per-class balance.
- **Accuracy/loss curves** over the 5 training epochs, showing both metrics improving steadily with a small validation loss spike around epoch 3 (right before the learning rate drop) before dropping sharply in epochs 4–5.
- **Confusion matrix heatmap** on the test set, which is essentially diagonal given the near-1.00 accuracy.
- A single **prediction example** at the end, where the model is run on a random held-out image and the predicted class is shown alongside the image and the true label.

These aren't saved to disk in the notebook, so they'd need to be exported and added to an `images/` (or similar) folder if you want them rendered on GitHub.

## Key Observations

- The model reaches over 99% validation and test accuracy within just 5 epochs, so this task turned out to be quite easy for a fairly small CNN — likely because rice grain images are visually simple (single object, clean background) and the dataset is large and perfectly balanced.
- The biggest jump in performance happens between epoch 3 and epoch 4, right after the learning rate was halved by `ReduceLROnPlateau`, suggesting the smaller learning rate helped the model settle into a better minimum.
- There's no sign of overfitting — training and validation accuracy stay close together throughout, and validation accuracy is actually slightly higher than training accuracy by the last epoch, likely because dropout is active only during training.
- All five classes get classified with similarly high precision/recall, so no rice variety stands out as harder to distinguish from the others.
- The `train_df`/`val_df`/`test_df` split created early in the notebook (via `train_test_split`) isn't actually the split used for training — the model is trained on a separate split produced by `image_dataset_from_directory`. Worth cleaning up if reusing this notebook.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
project/
│
├── rice-classification-cnn.ipynb
└── README.md
```

## How to Run

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Then launch Jupyter and open the notebook:

```bash
jupyter notebook
```

## Future Improvements

- Fix the unused `train_test_split` step so there's a single, consistent data split used throughout.
- Try more epochs with early stopping actually given a chance to trigger, to see where performance plateaus.
- Add data augmentation to test robustness to slightly different image conditions.
- Compare this custom CNN against a pretrained backbone (e.g. via transfer learning) to see whether it's actually needed given how well the small model already performs.
- Save training plots and the confusion matrix as image files in the repo so results are visible directly from the README.
