# Yikes... My data is shifting. Is my model still valid?

## TL;DR
* Methods are proposed to diagnose various types of dataset shift: prior, covariate, and concept shift.
* We apply these methods to the Payment Fraud dataset and find
  * Is there prior shift, i.e. $`P(Y)`$? **YES**, very clearly.
  * Is there (contitional) covariate shift, i.e.  ($`P(X|Y)`$) $`P(X)`$? **YES**, very clearly.
  * Is there concept shift, i.e. $`P(Y|X)`$? We tentatively
    * Propose that considering the leaf nodes of a tree based model leads to a compelling analysis of shifts in $`P(Y|X)`$.
    * Conclude the most clear signs of shifts in $`P(Y|X)`$ may come from general overfitting.

## Reminder of Dataset Shift
Simply put: the joint distribution $`P(X, Y)`$ may differ between the train and test/production data.

Here we consider classification, so
* $`X`$ is the feature vector
* $`Y`$ is the target class, e.g. fraud/non-fraud

Remember that from Bayes Rule we have the following factorisations
```math
P(X, Y) = P(Y|X) * P(X) = P(X|Y) * P(Y)
```  

This points to the various possibilites for shift, cf [our literature review](https://docs.google.com/presentation/d/1hsRrTQ5K19STH8jJ5P1idEyWHWBlg91lOWFHOLRMKas/edit#slide=id.g824c0f92da_0_575)
* $`P(X)`$ covariate: feature distributions 
* $`P(Y)`$ prior: e.g. fraud rate 
* $`P(Y|X)`$ concept: e.g. fraud rate in specific regions of the feature space
* $`P(X|Y)`$ condiction shift: feature distributions conditioned on the target (e.g. fraud vs non-fraud)

NB: it's difficult to imagine situations where shift happens in isolation. 
For example, a shift in $`P(Y)`$ must be accompanied by a shift in $`P(X)`$ 
and/or $`P(Y|X)`$, since $`P(Y) =  \sum_X P(Y|X) * P(X)`$. 
That said, some causal models may give rise to such scenarios, cf [our literature review](https://docs.google.com/presentation/d/1hsRrTQ5K19STH8jJ5P1idEyWHWBlg91lOWFHOLRMKas/edit#slide=id.g72c1b90b2a_0_47)

## How can you detect Dataset Shift?

Note that can treat whether the example comes from train/test as a (dummy) binary variable $`S`$ (for sample).
This means that we do the following.

### Prior shift: $`P(Y)`$
Let's start with the simplest case: shift in $`P(Y)`$. 

Shift is straightforward to establish: consider the contigency matrix of $`Y`$ and $`S`$, e.g. train/test vs fraud/non-fraud.
You can, for example, use [Pearsons chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test) to 
test whether there is a significant difference in $`P(Y)`$ between train and test, 
cf. this [code](project_utils.py#L230)/[test](tests/test_project_utils.py#L65).

||train|test|
|-|-|-|
|non-fraud | ... | ... |
|fraud | ... | ... |

### Concept shift: $`P(Y|X)`$

You can apply a similar method for $`P(Y|X)`$ with the caveats that 
1. It's not trivial how to condition on **regions in the feature space** $`X`$
2. We need to consider what we care about
  * Significant changes to $`P(Y|X)`$ between train and test? In this case we can use test as for $`P(Y)`$, namely the Pearsons chi-squared test
  * Whether a region in the feature space $`X`$ goes from fraud to non-fraud (or vice-verce), in other words: should the decision change? 
 
#### Conditioning on regions in the feature space $`X`$
One method that we propose is to consider the **leaf node assignments** based on tree-based models that predicts $`Y`$.
This method yields a contingency matrix of $`Y`$ and $`S`$ for every leaf node, cf this 
[code](leaf_node_analysis/spark.py#L11)/[test](tests/leaf_node_analysis/test_spark.py#L7).

We do want to point out that, although this is an interesting heuristic, we are not aware that it has any formal/theoretical guarantees.


#### Transitioning between Fraud vs Non-Fraud
We propose to consider the confidence intervals for the fraud rate for both test and train sets. 

For some target "precision" $`t_p`$ (e.g. 50%) we say that the region $`X`$ has trantitioned 
from Fraud to non-fraud if the full confidence interval for $`P(Y|X)`$ is completely
* above $`t_p`$ for the train set
* below $`t_p`$ for the test set

And vice-verca, cf. [the code](leaf_node_analysis/__init__.py#L8)/[tests](leaf_node_analysis/test__init__.py).

### Detecting shift in $`P(X)`$ and $`P(X|Y)`$
At first glance, it may seem daunting to compare entire feature distributions.
However, we do this all the time when we do **classification**.
If $`P(X|S=train)\ != P(X)\ != P(X|S=test)`$, then $`P(S|X) =  P(X|S) * P(S)/P(X)\ != P(S)`$. 
So we should be able to train a classifier that predicts $`S`$.

You can use your favorite evaluation metric (e.g. ROC-AUC or PR-AUC) to evaluate the "train/test classifier". 
If you not sure whether the result is **significant**, you can
* Simulate the null-hypothesis (e.g. create train/train samples with probabilities $`P(S)`$) 
* Establish standard error on your metric 
  * using bootstrapping 
  * use a closed form solution, cf. [this paper](https://pubs.rsna.org/doi/pdf/10.1148/radiology.143.1.7063747) for the standard error on the ROC-AUC \*

\* Cudo's to @lbernardi for finding this paper.

## Results for Payment Fraud

We used the methodology above using **Payment Fraud** setting. 
* **Data**: we used the control group, which is an unbiased sample of all transactions that was delibarately not blocked, cf. [this code](project_utils.py#L13)
* **Train/Test** periods, see image below
  * There is typically a **gap of 3 months** between **train** and production (i.e. **test**) data sets, reflecting the time that fraud labels mature.
  * We consider a training period of (roughly) 9 months
  * To **mimic weekly retraining**, we calculate a specific **train period** for each **one week of test data** like in , cf. [this code](propensity_utils.py#L16)
  * In most of the analyses below we considered multiple test weeks over a period of time, 
  cf. this [code](2020-09-cost-of-fraud/project_utils.py#L223)/[test](tests/test_project_utils.py#L47).

![weekly_retraining](plots/weekly_retraining.png "Weekly Retraining")


### Prior shift: $`P(Y)`$

In the figure below the <span style="color:blue">blue line (blue shaded area)</span> shows how the fraud rate ($`90\%`$ CL interval) changes per test week. 

We marked in <span style="color:red">red</span> the test weeks for which $`P(Y)`$ is significantly different between train and test, specifically: the p-value of the Pearson's Chi2 test $`\leq 5\%`$.

The analysis was done in [this notebook](2020_10_09_1_diagnosis_and_comparison.ipynb). 

![prior shift](plots/prior_shift.png "Prior Shift")


#### Covariate Shift: $`P(X)`$

As explained [above](#detecting-shift-in-px-and-pxy) we detect covariate shift using classifiers that predict on 
the dummy variable $`S`$ reflecting whether an example is in the train or test set.
Specifically, we use the train/test split setup [mimicing weekly retraining](#results-for-payment-fraud).

In [this notebook](2020_12_10_1_december_focus_days.ipynb) we
* Trained the train/test classifier (random forests: `nfolds=5, max_depth=5, ntrees=50`) for test weeks between December 2019 and September 2020
* Looked at their performance (ROC-AUC, PR-AUC) as well as the feature importances

The train/test classifier is has ROC-AUCs in the high 90's:

![train_test_classifier_roc_auc](plots/train_test_classifier_roc_auc.png "Train/Test ROC-AUCs")

Since the train/test classifier performs so well we don't need a statistical test to establish **Covariate Shift**.

When we look at the featuers with the hightest importances we indeed see that you'd expect to shift
* "Time since" feautres, that only increment over time: property/country pbb age, time since first/second ok booking
* Probability features $`P(... |\text{country})`$

![train_test_classifier_importances](plots/train_test_classifier_importances.png "Train/Test Feature Importances")

In [the notebook](2020_12_10_1_december_focus_days.ipynb) more properties of the covariate shift are studied:
* Single feature train/test classifiers
* Effect of taking out features from the train/test classifier

In [this notebook](2021_01_15_3_P_X_given_Y.ipynb) we also considered $`P(X|Y)`$. 
The picture is very similar, namely: very clear shift.


### Concept shift: $`P(Y|X)`$

We analysed **one test week (starting Monday 2019-12-2)**.

In [this notebook](leaf_node_based_p_Y_given_X__prepare_leaf_node_file__analyse_pvalues.ipynb) we
* Trained a XGBoost fraud classifier (PR-AUC train: 74.2%, train: 53.7%)
* Got the leaf nodes assignments for the train and test sets (3154 leaf nodes in total)
* Calculated the contigency matrix (fraud/non-fraud vs train/test) per leaf node
  * Calculated the corresponding p-value per leaf node

#### P-value analysis
Here you see the overall p-value distribution for the 3154 leaf nodes.

![leaf_node_p_value_distribution](plots/leaf_node_p_value_distribution.png "P-value Distribution")

Let's examine the leaf nodes with the lowest p-values. 
Whilst $`P(Y|X)`$ differs significantly between train and test, we see that the fraud rates in both train and test are very low.

![leaf_node_table_lowest_p_value](plots/leaf_node_table_lowest_p_value.png "P-value Distribution")

For more analysis we refer to [the notebook](leaf_node_based_p_Y_given_X__prepare_leaf_node_file__analyse_pvalues.ipynb), 
though it appears that examining significant changes may not be very usefull.

#### Transitioning between Fraud vs Non-Fraud

##### One week (starting Monday 2019-12-2)
In [this notebook](leaf_node_based_p_Y_given_X__investigate.ipynb) we investigate whether any leaf nodes trasitioned as explained above.
We consider
* Confidence levels 
   * 5%: the usual choice
   * 0.1%: attempt to account for calculating 3154 intervals respectively\*. 
* Target "precisions" $`t_p`$ of
  * 50%: the most natural (?) arbitrary choice
  * 15%: reflecting where you would rougly place the decision boundary in Payment Fraud, 
  since naively a True Positive costs the TTV, whilst a false postivie costs the commission of 15%

\* Question: should we use 5%/3154 instead? In this case it does not matter for the results?

For the 5% confidence level and target "precision" $`t_p=50\%`$  we find the following "transitions" (all from fraud to non-fraud)

![target_precision_50_alpha5](plots/target_precision_50_alpha5.png "50% Target Precision 5% CL")

Whilst target "precision" $`t_p=15\%`$  we find (all from non-fraud to fraud)

![target_precision_15_alpha5](plots/target_precision_15_alpha5.png "15% Target Precision 5% CL")

Interestingly, when we use a confidence level of 0.1% we don't find any transitions.

These perhaps raise more questions than answers:
* I the approach valid in the first place? 
  * Do we need to take into account that the trees were trained to separate fraud/non-fraud?
  * We already know that the XGBoost model is overitted based on the PR-AUC (train: 74.2%, train: 53.7%)
* If the approach is valid, then what should be the confidence level?
  * Every datapoint is double counted as many times are there are trees. 
  * There are likely correleations among leaf nodes?
* **If a confidence level of $`5%/n_{leaves}`$ is valid, can we then conclude that there was no significant concept shift?**
  * Should we extend the analysis to more weeks of test data? Currently we only have the week starting Monday 2019-12-2.

##### Multiple weeks
To answer the last question we extended that analysis to the full range of test weeks from 2019-12-02 to 2020-9-14, cf. [this notebook](2021_01_15_2_analyse_P_X_given_Y_over_time.ipynb).

Like before we consider
* Confidence levels  5%, 0.1%
* Target "precisions" $`t_p`$ of 50% and 15% 

We find the following picture for 0.1% confidence level:

![p_y_given_x_over_time](plots/p_y_given_x_over_time.png "P(Y|X) over time")

We note that
* For  $`t_p`$ of 50%, marked in blue, the most significant changes appears to be in the first few months of test weeks. 
Note that the corresponding training periods have large overlap with the training period for the leaf nodes (for the first test week the training period is the same by definition). Hence we believe that the "concept shift" that is highlighted is mostly because of "overfitting".
* For $`t_p`$ of 15%, marked in organge, there seems to be an interesting peak in April. Although this does coincide with the period of large covariate shift $`P(X)`$ and prior shift $`P(Y)`$, inspection of the data reveals that most of these leave nodes have less than 20 data points in the test set. Therefore we deam these results less reliable.


## Conclusion

From all the above we conclude
* Is there prior shift, i.e. $`P(Y)`$? **YES**, very clearly.
* Is there (contitional) covariate shift, i.e.  ($`P(X|Y)`$) $`P(X)`$? **YES**, very clearly.
* Is there concept shift, i.e. $`P(Y|X)`$? We tentatively
  * Propose that considering the leaf nodes of a tree based model leads to a compelling analysis of shifts in $`P(Y|X)`$.
  * Conclude the most clear signs of shifts in $`P(Y|X)`$ may come from general overfitting.
