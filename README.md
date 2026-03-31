# mnist-naive-bayes-from-scratch
Implemented Discrete and Gaussian Naive Bayes from scratch for MNIST digit classification, with posterior analysis and model comparison.



# MNIST Digit Classification with Naive Bayes from Scratch

This project implements **Naive Bayes from scratch** for handwritten digit classification on the **MNIST dataset**.  
Two variants were built and compared:

- **Discrete Naive Bayes**
- **Continuous Naive Bayes (Gaussian Naive Bayes)**

The project focuses on understanding how different probabilistic assumptions affect classification performance on image data.  
In the final evaluation, the **discrete model achieved 84.68% accuracy**, while the **continuous model achieved 68.20% accuracy** on the MNIST test set.

---

## Overview

Handwritten digit recognition is a classic machine learning problem in which grayscale images of digits from **0 to 9** must be classified correctly.

In this project, I implemented Naive Bayes without using machine learning libraries for model training. The goal was not only to predict the correct digit label, but also to understand:

- probabilistic classification
- prior and likelihood estimation
- Laplace smoothing
- Gaussian feature modeling
- the effect of feature representation on model performance

---

## Dataset

This project uses the **MNIST handwritten digits dataset**.

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Image size:** 28 × 28 pixels
- **Feature size:** 784 pixels per image
- **Classes:** digits 0 through 9

The dataset is read directly from the standard MNIST binary files (`idx` format).

---

## Methodology

### 1. Prior Computation

For each digit class, prior probabilities are computed from the training labels.  
To avoid zero probabilities, **add-one smoothing** is applied.

---

### 2. Discrete Naive Bayes

In the discrete version, every pixel intensity is converted into one of **32 bins** by grouping values from 0 to 255 into fixed intervals.

#### Steps
- Count how often each pixel falls into each bin for every digit class
- Apply **Laplace smoothing**
- Convert probabilities to **log space** for numerical stability
- Predict the class with the highest posterior score

#### Why this works well
MNIST digits are sparse and structured. Many pixels are either close to background or stroke intensity, so binning captures these patterns effectively.

---

### 3. Continuous Naive Bayes

In the continuous version, each pixel is modeled using a **Gaussian distribution** for each class.

#### Steps
- Compute the **mean** of each pixel for every digit class
- Compute the **variance** of each pixel for every digit class
- Use the Gaussian probability density function to compute likelihoods
- Predict the class with the highest posterior score

#### Stability handling
Very small variances can make Gaussian likelihoods unstable, so a minimum variance threshold is used.

#### Limitation
This approach assumes pixel intensities follow a Gaussian distribution, which is often not a strong fit for MNIST image data.

---

## Results

### Final Performance

- **Discrete Naive Bayes**
  - Test images: 10,000
  - Errors: 1,532
  - Accuracy: **84.68%**

- **Continuous Naive Bayes**
  - Test images: 10,000
  - Errors: 3,180
  - Accuracy: **68.20%**

### Key Insight

The discrete model significantly outperformed the continuous model.  
This suggests that for MNIST, **binning pixel intensities is more effective than assuming a Gaussian distribution for each pixel**.

---

## What I Learned

This project helped me strengthen my understanding of:

- Naive Bayes from first principles
- log-probability based inference
- Laplace smoothing
- Gaussian likelihood estimation
- numerical stability in probabilistic models
- how modeling assumptions directly affect real-world performance

The biggest takeaway was that **the choice of feature representation matters as much as the classifier itself**. Even with the same Naive Bayes framework, changing the way pixel values are modeled produced a large difference in accuracy.

---

## Sample Output

The program prints:

- posterior probabilities for each test image
- predicted label vs actual label
- final evaluation metrics
- an “imagination” of each digit learned by the classifier

Example summary:

```text
Discrete Naive Bayes
Total test images: 10000
Errors: 1532
Error rate: 0.1532
Accuracy: 0.8468






## Dataset Setup

This project uses the **MNIST handwritten digits dataset**.

Download the following four files and place them inside a local `data/` folder:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Your folder structure should look like this:

```text
mnist-naive-bayes/
│── data/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx1-ubyte
│   └── t10k-labels.idx1-ubyte
│── main.py
│── README.md
