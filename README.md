# Project: Sign Language Recognition System
## Artifical Intelligence: Hidden Markov Model & N-gram Language Model

## Overview
This project is a partial requirement of Artificial Intelligence Nano Degree at Udacity. In this project, a system that can recognize words from the American Sign Language (ASL) is built. A preprocessed dataset of tracked hand and nose positions extracted from video is provided. First of all, this dataset is processed to generate different feature sets. Next, a set of Hidden Markov Models (HMMs) using different set of features are trained. The number of states in HMMs is a tunning parameter. Different values of the number of states are attempted using three different criteria: Cross-Validation, BIC and DIC. For each criterion, the optimal number of states is selected. This models are tested on the test data set. The evaluation metrics used is word error rate (WER). The results showed that when the features are polar coordinate values where the nose is the origin and the optimal number of states is selected based on BIC, the WER is 56.2%. 

The WER is expected be further improved by introducing N-gram Statistical Language Models, which capture the  the conditional probability of particular sequences of words occurring. First of all, a set of candidate predictions are made by HMMs. Then, we N-gram Statistical Language Models to select the one with the highest conditional probability. Different "N" in N-grame are attempted. This part is still under construction.

## Software Requirement
This project requires Python 3 and the following Python libraries installed:

1. NumPy
2. SciPy
3. scikit-learn
4. pandas
6. matplotlib
6. hmmlearn

## Code
1. asl_recognizer.ipynb: notebook contains top-level code, explanation, results and structure of the project
2. my_model_selectors.py: implementation of 3 model selection criteria: Cross-Validation, BIC and DIC
3. my_recognizer.py: implementation of predicting function
4. asl_test.py: supporting code for testing
5. asl_utils.py: supporting code
6. asl_data.py: supporting code for data pre-processing
