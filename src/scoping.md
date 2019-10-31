# Fraud Scope

1. What preprocessing might you want to do? How will you build your feature matrix? What different ideas do you have?
  * We will have an emphasis on binary encoded predictors and normalized numerical predictors with choice one-hot encoded columns where appropriate.
  * We also plan to create some 'is-null' features for whether some fields are populated or not (i.e. venue address, payee name)

2. What models do you want to try?
  * We will use logistic regression as a starting baseline model, but we plan to also test out a decision tree, random forest, and boosted models. 

3. What metric will you use to determine success?
  * NOT accuracy
  * We need to minimize both false positive and false negative, so F1 will be our target score.
  * We will also consider recall as a target score because our model results are intended to be reviewed by a human to limit potential false positives.