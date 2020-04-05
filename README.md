# CREDIT RISK TASK

> German Credit data This dataset classifies people described by a set of attributes as good or bad credit risks.

- data distribution: bad:good (3:7) unblanaced data

- data source: https://www.openml.org/d/31

**It is worse to class a customer as good when they are bad, than it is to class a customer as bad when they are good.**

## Process:

### data propressing
- data understanding
- data analysis (statistic, PCA)
- feature engineering (one hot encoding, label encoding, standardization)

### model constriction

- Evaluation Score: 0.6 * Bad recall + 0.4 * weight_F1
- Approch : 10 cross validation 
- Unbalanced Data Issue: class weight, upsampling
- Model
    - AdaBoost
    - Random forest
    - Support Vector Machine
    - Deep Neural Network

### Evaluation Score

- SVM: 0.661
- AdaBoost: 0.710
- random forest: 0.795
- DNN: 0.922

