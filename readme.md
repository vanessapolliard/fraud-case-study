# Real-Time E-Commerce Fraud Screening

## Goals
Our goal is to create a sustainable, cloud-based software project to flag potential fraudulent events posted to the company software platform in real-time.

We were given access to a data set consisting of ~14,000 events which were already classified as being legitimate or fraudulent. There were ~13,000 legitimate and ~1,300 fraudulent events.

## Process Flow
We placed a heavy emphasis on mapping out this project and planning out our next steps every few hours. Whiteboards are our friends.

<p align="center">
<img src="images/board_img_1.jpg" width="600">
</p>  

![](images/board_img_2.jpg)

Instead of splitting up this project and working on sections individually, we mostly employed a 3-way group programming approach. We alternated projecting one of our screens on a TV and worked together to solve issues. We found that we all learned new things while working this way, and we were able to quickly resolve issues using our collective knowledge. When appropriate, we also split up and worked individually on sections of the project, while checking in frequently.

## Data Preprocessing
### Restricting Available Data
Our initial data set contained a number of pieces of information that would not generally be know at the time of creation of an event, so we removed them. These included:
  * Info pertaining to payouts for the event in question.
  * Ticket sales info.
  * Payee info.
  * Time-based info that would not be available when the event was created.

### Exploratory Data Analysis
We made used of several helper functions that we created while initially exploring the data. These functions allowed us to quickly compare values for a given predictor for legitimate and fraudulent events. Our initial focus was on finding easy predictors for our model, which included:
  * Features that were binary encoded (e.g. "has_logo").
  * Numeric features, which we normalized (e.g. "body_length").
  * Easily extracted numerical features (e.g. number of "previous_payouts").

**Example helper function output:**
```python
mean_comparison('user_age')

user_age fraud mean:
87.15

user_age not fraud mean:
402.68
```


### Feature Engineering
In order to easily experiment with various preprocessing steps and model configurations, we utilized the `sklearn.pipeline.Pipeline` class. Doing this, we were able to chain together cleaning / processing steps and different models:

1. Process numeric columns, including normalization.
2. Process categorical columns, including one-hot encoding.
3. Fit a model.

### Imbalanced Class
Our initial exploratory analysis showed that our data set consisted of ~90% legitimate events and ~10% fraudulent events. Since all of our models are sensitive to imbalanced classes while training, we utilized the `imbalanced-learn RandomOverSampler` tool to oversample our fraudulent events, resulting in a balanced class.


## Modeling
### Model Accuracy Metrics
Our model accuracy metrics were chosen with the intended purpose of this tool in mind, which is to flag events that are **potentially** positive for fraud, so they can then be reviewed by an actual human being.

  * As such, our goal was to minimize both false positive and false negative results, so an F1 score was our target.
  * We also considered recall as a target score, because review by a human will limit the impact of events falsely flagged as fraudulent.

### Model Iteration Process
To aid in our model creation and selection process, we stored the parameters and metrics for each model we ran in a pandas DataFrame.

| id | filename | pipeline_string | named_steps | threshold | accuracy | precision | recall | auc | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 601551091133794381 | /Users/ste | Pipeline(... | {'ct': ... | 0.5 | 0.98 | 0.95 | 0.87 | 0.99 | 0.91 |
| 7331089530306465686 | /Users/ste | Pipeline(... | {'ct': ... | 0.5 | 0.98 | 0.93 | 0.85 | 0.99 | 0.89 |
| 2989737759052901380 | /Users/ste | Pipeline(... | {'ct': ... | 0.5 | 0.98 | 0.95 | 0.86 | 0.99 | 0.9 |
| TEST | /Users/ste | Pipeline(... | {'ct': ... | 0.5 | 0.98 | 0.95 | 0.84 | 0.99 | 0.89 |

### Validation And Testing Methodology
In order to be as thorough as possible, we performed a stratified split of our data at the very beginning, in order to have a holdout data set for final model accuracy testing.

While selecting and tuning our models, we employed a GridSearchCV strategy, which varies specified model parameters while performing cross-validation using our testing data.

## Results

### ROC Curve
This chart illustrates the diagnostic ability of our binary classifier system as its discrimination threshold is varied.
<p align="center">
<img src="images/roc_curve_final_model.png" width="600">
</p>  

### Model Metrics
| Metric | Score |
| --- | --- |
| classification_threshold | 0.5 |
| cv_accuracy | 0.982 |
| cv_precision | 0.954 |
| cv_recall | 0.836 |
| cv_roc_auc | 0.986 |
| cv_f1 | 0.891 |

### Tuning Thresholds
| Classification Threshold | Recall |
| --- | --- |
| 0.5 | 0.879 |
| 0.4 | 0.898 |
| 0.2 | 0.947 |
| 0.1 | 0.978 |

### Top Model Feature Importances
| Importance | Feature |
| --- | --- |
| 0.31 | num_previous_payouts |
| 0.11 | user_age |
| 0.06 | body_length |
| 0.06 | time_to_create |
| 0.06 | user_type_1.0 |
| 0.05 | org_twitter |


## Cloud-Based Web App
Our trained, pickled model was deployed to an AWS instance. 

## App Link: http://bit.ly/fraud_predictor

## Future Steps
While we are satisfied with the progress we've made so far, given additional funding for this project, our future goals would include the following:
  * Additional feature engineering, including Natural Language Processing of free-form predictor information.
  * Additional model types and tuning.