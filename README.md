# Face Recognition using PCA and Softmax Classifier

This repository contains a project for **face recognition and classification** using Principal Component Analysis (PCA) and a custom Softmax PCA Classifier. The implementation focuses on dimensionality reduction, image reconstruction, and classification tasks based on the AT&T Face Dataset. Note for the **assignment submission**, you just need to run `q1.py` and `q3_bonus.py`. Codes for q2 is in the Matlab folder.

---

## Overview

This project demonstrates two major tasks:

1. **Dimensionality Reduction using PCA**:
   - Extracts **eigenfaces** as principal components to represent face images in a lower-dimensional space.
   - Performs face **reconstruction** using a varying number of principal components.
2. **Face Classification with a Softmax PCA Classifier**:
   - PCA-reduced features are used as input to a **Softmax classifier** for multi-class classification.
   - The model is evaluated on known and unknown subjects, as well as non-face and modified-face images.

The project uses the **AT&T Face Dataset**, consisting of 400 grayscale images (40 subjects, 10 images each), each of size 112x92 pixels.

---

## Features

- **Eigenface Visualization**: Visualize the most significant eigenfaces that capture facial variations.
- **Face Reconstruction**: Reconstruct faces using different numbers of eigenfaces to analyze reconstruction quality.
- **Softmax PCA Classifier**:
  - Combines PCA for dimensionality reduction with a Softmax classifier for face recognition.
  - Provides probabilistic outputs and confidence scores.
- **Evaluation**:
  - Accuracy on training, known test, and unknown test subjects.
  - Non-face rejection rate and robustness to modified images.
  - Confidence analysis to interpret prediction certainty.
- **Visualization**: Generate visualizations of predictions, confidence distributions, and reconstruction results.

---

## Dependencies

The project requires the following libraries:

- Python 3.x
- NumPy
- Matplotlib
- Pillow (PIL)
- OS and Random libraries (built-in)

To install the required dependencies, run:

```bash
pip install numpy matplotlib pillow
```

---

## Code Files

### 1. `q1.py`: Eigenface Extraction and Face Reconstruction

This script performs the following tasks:

- Loads and processes the AT&T Face Dataset.
- Computes eigenfaces using **PCA** via Singular Value Decomposition (SVD).
- Visualizes the top eigenfaces.
- Reconstructs face images using different numbers of principal components.
- Outputs key visualizations:
  - **eigenfaces.png**: Visualization of leading eigenfaces.
  - **variance_explained.png**: Cumulative explained variance plot.
  - **reconstructions.png**: Face reconstruction with varying numbers of components.

### 2. `q3_bonus.py`: Softmax PCA Classifier for Face Recognition

This script introduces a new classification method:

- **Softmax PCA Classifier**:
  - Reduces dimensionality using PCA.
  - Trains a Softmax classifier for multi-class face classification.
  - Implements gradient descent with early stopping and L2 regularization.
- **Evaluation**:
  - Accuracy for training, known test subjects, and unknown test subjects.
  - Non-face rejection rate and modified face detection rate.
  - Confidence score analysis.
- **Visualization**:
  - Generates visual outputs for predictions with confidence scores and probability distributions.

Outputs include:

- **eigenfaces.png**
- **variance_explained.png**
- **reconstructions.png**
- Predictions visualized in the `output/` directory.

---

## How to Run

### Step 1: Prepare the Dataset

1. Download the **AT&T Face Dataset** [here](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).
2. Place the dataset in the following structure:

   ```bash
   dataset/
       s1/
           1.pgm
           2.pgm
           ...
       s2/
           1.pgm
           ...
       ...
       s40/
   ```

3. Create directories for **non-face images** and **modified face images** if applicable:

   ```bash
   non_face_images/
   modified_faces/
   ```

### Step 2: Run `q1.py` (Eigenfaces and Reconstruction)

Execute the script to perform PCA, visualize eigenfaces, and reconstruct face images:

```bash
python q1.py
```

### Step 3: Run `q3_bonus.py` (Softmax PCA Classifier)

Train and evaluate the Softmax PCA Classifier on the AT&T Face Dataset and additional test sets:

```bash
python q3_bonus.py
```

Results, including accuracy and prediction visualizations, will be saved in the `output/` directory.

---

## Results

### Classification Results

| Metric                        | Value |
|-------------------------------|-------|
| Training Accuracy             | 1.000 |
| Test Accuracy (Known Subjects)| 1.000 |
| Test Accuracy (Unknown Subjects)| 0.000 |
| Non-Face Rejection Rate       | 0.200 |
| Modified Face Detection Rate  | 1.000 |

### Confidence Statistics

| Dataset             | Mean Confidence |
|----------------------|-----------------|
| Training Images      | 0.996           |
| Test Known Images    | 0.968           |
| Test Unknown Images  | 0.526           |
| Non-Face Images      | 0.822           |
| Modified Images      | 0.998           |

### Visual Outputs

1. **Eigenfaces**: Top eigenfaces visualized.
2. **Reconstruction**: Face images reconstructed with increasing components.
3. **Classification Predictions**:
   - Predicted class labels with confidence scores.
   - Top-5 probability distributions for each sample.

---

## Key Insights

1. **PCA Effectiveness**:
   - PCA successfully reduces dimensionality while retaining key facial features.
   - Around 300-350 principal components capture over 99\% of the variance.

2. **Softmax PCA Classifier**:
   - Achieves perfect accuracy for training data and known test subjects.
   - Struggles to generalize to unseen subjects (accuracy = 0.000), highlighting limitations for open-set recognition.
   - Robust to minor modifications in face images.

3. **Non-Face Rejection**:
   - The model has limited ability to reject non-face inputs, indicating a need for better open-set recognition mechanisms.

---

## Acknowledgements

- The AT&T Face Dataset was provided by the [Cambridge University Computer Laboratory](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).
- Implementations rely on standard Python libraries like NumPy and Matplotlib.
- This README is generated by ChatGPT-4o.
