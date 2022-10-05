# Credit_Risk_Analysis
Analyzing credit risk using algorithms (RandomOverSampler, SMOTE, ClusterCentroids, and SMOTENN) and Machine Learning models (BalancedRandomForestClassifier and EasyEnsembleClassifier).


## Overview
Below is an analysis of using various supervised machine learning models to assess credit card risk using a dataset from LendingClub.  Various algorithms were used to both oversample and undersample the data to try to reduce bias in this unbalanced dataset.  The precision, reacall, and balanced accuracy scores were then used to determine if any of these models are a good fit to use to predict future credit card risk.

### Technology/languages Used
Jupyter Notebook
Python -imbalanced-learn and scikit-learn libraries

## Results
### Naive Random Oversampling
![naive_random_oversamplong](https://user-images.githubusercontent.com/102555125/194111826-8c27ff38-98d5-4d7c-8474-b8e90699ed4e.png)

- **_Balanced Accuracy Score: 0.64_**
- Low-risk Precision: 1.00
- High-risk Precision: 0.01
- Low-risk Recall: 0.65
- High-risk Recall: 0.62

### SMOTE Oversampling
![SMOTE_oversampling](https://user-images.githubusercontent.com/102555125/194111863-52e5cf8f-0bbb-4670-93dd-9d25dd767620.png)

- **_Balanced Accuracy Score: 0.63_**
- Low-risk Precision: 1.00
- High-risk Precision: 0.01
- Low-risk Recall: 0.64
- High-risk Recall: 0.62

### Cluster Centroids Undersampling
![ClusterCentroids_undersampling](https://user-images.githubusercontent.com/102555125/194111905-886004f5-2d0d-49b8-9a26-404f13a21d75.png)

- **_Balanced Accuracy Score: 0.51_**
- Low-risk Precision: 1.00
- High-risk Precision: 0.01
- Low-risk Recall: 0.43
- High-risk Recall: 0.59

### SMOTEENN Combination Sampling
![SMOTEENN_combosampling](https://user-images.githubusercontent.com/102555125/194111922-f991cbf7-7fea-4d29-814d-98899f3685e0.png)

- **_Balanced Accuracy Score: 0.65_**
- Low-risk Precision: 1.00
- High-risk Precision: 0.01
- Low-risk Recall: 0.59
- High-risk Recall: 0.71

### Balanced Random Forest Classifier
![BalancedRandomForest_ensemble](https://user-images.githubusercontent.com/102555125/194111937-698815a0-8bf7-4b57-9899-edeca7a9e0df.png)

- **_Balanced Accuracy Score: 0.67_**
- Low-risk Precision: 1.0
- High-risk Precision: 0.73
- Low-risk Recall: 1.0
- High-risk Recall: 0.34

### Easy Ensemble AdaBoost Classifier
![Easy_Ensemble](https://user-images.githubusercontent.com/102555125/194111973-413c19ab-6596-49ea-900f-24291384b6d3.png)

- **_Balanced Accuracy Score: 0.67_**
- Low-risk Precision: 1.00
- High-risk Precision: 0.07
- Low-risk Recall: 0.94
- High-risk Recall: 0.91

## Summary
None of the six models had particularly impressive balanced accuracy scores as none of them even broke the 0.70 mark -so none of them performed great overall.  In assessing credit risk, it is often more important to correctly identify which prospects are more likely to default on their credit card (high-risk).  For this reason, a model that has high scores in both precision and recall for high-risk prospects is necessary.  Unfortunately, none of these models performed well enough in correctly identifying the high-risk prospects to use on a regular basis.  The best performing model for high risk was the Balanced Random Forest Classifier, and at only 0.34 scored on recall, I wouldn't trust it without reservations.

Overall, my recommendation would be to keep tweaking models to find one that scores better on balanced accuracy while maintaining high precision AND recall scores in the high-risk category.  If that is not an option, using the Balanced Random Forest Classifier is going to weed out more high-risk credit card prospects than the others.
