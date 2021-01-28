# Predict tags on StackOverflow with linear models

## Table of Contents
* Text Preprocessing & Tokenizing

* Transforming text to a vector in 2 methods
** bags of words
** TfidfVectorizer from scikit-learn

* MultiLabel classifier
** Logistic Regression as the basic classifier
** OneVsRestClassifier used on top of the Logistic Regression for k classes

* Performance Evaluation
** Accuracy
** F1-score (F1-score is used for final evaluation since the k classes are unbalanced)
** ROC curve

* Model Optimization & Regularization

* Feature Importance Analysis

