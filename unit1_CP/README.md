# CP1 - Unigram Probabilities

Instructions for this assignment are here:

https://www.cs.tufts.edu/cs/136/2023s/cp1.html

# Install

Following the instructions here for the standard "spr_2021s_env" used throughout this course:

https://www.cs.tufts.edu/cs/136/2023s/setup_python_env.html

# Data Analysis Task

We're analyzing the text observed in the annual State of the Union speech by the US President, with a dataset going back to 1945.

* Train on speeches from 1945 - 2015 (almost 400,000 words!)
* Test on the speech in 2016

We'll assume that we have a known vocabulary with a fixed set of V possible words, known in advance.

The two problems in this CP will address two key questions:

* 1) Given training data, how should we estimate the probability of each word appearing again at test time? How might estimates change if we have very little (or abundant) data?

* 2) How can we select hyperparameter values to improve our predictions on heldout data, using only the training set?


# Outline

Your goal is to build an estimator for predicting a vocabulary word's probability of appearing in new data, based on a training dataset of many words.

Your task in Problem 1 is to implement 3 kinds of estimators in python code and check their results on a test dataset.

* [ ] Maximum likelihood - See `MLEstimator.py`
* [ ] Maximum a posteriori - See `MAPEstimator.py`
* [ ] Posterior predictive - See `PosteriorPredictiveEstimator.py`

Your task in Problem 2 is to learn how to select the *hyperparameter* $\alpha$ that controls the Posterior Predictive estimator.

* [ ] Implement `calc_per_word_log_evidence` in `run_select_alpha.py`
* [ ] Implement `main` in `run_select_alpha.py`
