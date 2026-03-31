# mnist-naive-bayes-from-scratch
Implemented Discrete and Gaussian Naive Bayes from scratch for MNIST digit classification, with posterior analysis and model comparison.


Handwritten Digit Classification using Naive Bayes

This project implements Naive Bayes from scratch for handwritten digit classification on the MNIST dataset.
I built and compared two variants:

Discrete Naive Bayes
Continuous Naive Bayes (Gaussian Naive Bayes)

The goal was not just to classify digits, but also to understand how different modeling assumptions affect performance on image data.


Problem Statement

Handwritten digit recognition is a classic machine learning problem where the task is to classify grayscale images of digits from 0 to 9.

In this project, I implemented Naive Bayes classifiers from scratch to predict the correct digit label for each image. I also compared two different ways of modeling pixel values:

treating pixel intensities as discrete bins
treating pixel intensities as continuous Gaussian variables

This project helped me understand:

probabilistic classification
class priors and likelihoods
Laplace smoothing
Gaussian likelihood estimation
how feature representation impacts model performance

Dataset Used

This project uses the MNIST handwritten digits dataset.

Each image is 28 × 28 pixels
Each image is grayscale
Each image is flattened into a 784-dimensional feature vector
Labels range from 0 to 9
The evaluation was done on 10,000 test images

The dataset files are read directly from the standard MNIST binary files:

training images
training labels
test images
test labels


Approach
1. Prior Probabilities

I first computed the prior probability of each digit class based on the training labels.
To avoid zero probabilities, I applied add-one smoothing when calculating priors.

2. Discrete Naive Bayes

In the discrete version, each pixel value is converted into one of 32 bins by dividing the original intensity range 0–255 into groups.
This means each pixel is treated as a categorical variable instead of a continuous value.

Steps
For each class (0–9), count how often each pixel falls into each of the 32 bins
Apply Laplace smoothing to avoid zero likelihoods
Compute log probabilities for numerical stability
For a test image, sum:
log prior
log likelihood of every pixel bin given the class
Why this works well

MNIST pixel values are not truly smooth continuous measurements in practice. Many pixels are either near black or near white, so binning helps capture this pattern better.

3. Continuous Naive Bayes

In the continuous version, each pixel is modeled using a Gaussian distribution for each class.

For every digit class and every pixel:

compute the mean
compute the variance
use the Gaussian probability density function during prediction

A variance floor was applied in the implementation so that very small variances do not cause instability.

Steps
Estimate per-class, per-pixel mean
Estimate per-class, per-pixel variance
Use Gaussian likelihoods for all 784 pixels
Add log prior and log likelihoods to compute class score
Limitation

This method assumes pixel intensities follow a Gaussian distribution, which is often a poor fit for MNIST. That is one major reason why this model performed worse than the discrete version.

Results
Final Accuracy Comparison
Model	Test Images	Errors	Accuracy
Discrete Naive Bayes	10,000	1,532	84.68%
Continuous Naive Bayes	10,000	3,180	68.20%



How to Run : 
Prerequisites
Python 3
NumPy
Files needed : 
You need the MNIST binary files:

train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
Run modes

The project supports two modes:
0 → Discrete Naive Bayes
1 → Continuous Naive Bayes

Example
python naive_bayes.py 0
python naive_bayes.py 1


