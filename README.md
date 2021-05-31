# Kaggle 2019 Data Science Bowl: 143th place (top 5%) silver medal solution
Members:
* [Artsem Zhyvalkouski](https://www.kaggle.com/aruchomu)
* [Maxim Gritsenia](https://www.kaggle.com/maximgritsenia)

Here is a writeup of our solution which scored QWK(quadratic weighted kappa) of 0.544 on private leaderboard, reached 143th place among 3497 teams and brought us our first silver medal.
![](https://www.gpb.org/sites/www.gpb.org/files/styles/redesign_800x450/public/blogs/images/2016/08/04/from_press_release_pbs_measure_up.jpeg?itok=RxJZpdQ6)

## 1. Preprocessing and Feature Engineering
For each assessment trial we used the whole previous history of a user to extract features in an accumulative manner. We also took advantage of the test data and created training samples using previous assessments. Note: 'time' in the list below refers to seconds. 
### 1.1. Counter/accumulative features
In addition to raw counters, most of the counter features were also normalized by 'number of events' / 'time spent in game' and added as additional features.
* Session type counter
* Changing session type counter
* Session title counter
* Event code counter
* Title + event code interaction counter
* Accuracy group counter
* Total sessions counter
* Total events counter
* Total time for each session title
* Session world counter
* Total time for each world
* Total 'Game' time for each world
* Total 'Activity' time for each world
* 'misses' counter for all previous sessions
* 'correct' counter for all previous sessions
* 'incorrect' counter for all previous sessions
* Weekday counter for each session
* Hours counter for each session
* Current assessment title counter
* Current assessment world counter
* Current assessment title total time
* Current assessment world total time
* Current assessment world total 'Game' time
* Current assessment world total 'Activity' time

### 1.2. Numerical sequence statistics 
For the following sequences we calculated various statistics such as: mean, median, sum, max, min, std, skew, lag, difference of lag_1 and lag_2.
* Number of correct attempts in previous assessments
* Number of incorrect attempts in previous assessments
* Accuracy in previous assessments
* Accuracy group in previous assessments
* Durations of previous assessments
* Durations of previous sessions
* Number of events in previous sessions

### 1.3. Categorical features
All the categorical features were encoded using integers. We used LGBM encoder for some of them (described in section 3).
* Assessment title
* Assessment world <br>
For the following sequences we calculated: mode, number of unique categories, lag category.
* Previous assessment titles
* Previous session titles

### 1.4. Other features
* Last accuracy for each assessment title
* Last accuracy group for each assessment title
* Seconds since installation
* Days since installation
* sin/cos of hour, day, weekday
* Mean time spent per day
* Current assessment title mean time
* Current assessment world mean time
* Mean of last accuracies for all assessment titles
* Ratio of life spent in game (time spent in game / time since installation)
* Time spent per session (time spent in game / number of sessions)
* Number of sessions per day (number of sessions / days since installation)
* Number of events per session (number of events / number of sessions)
* Time spent per event (time spent in game / number of events)
* Number of events per day (number of events / days since installation)
Note: due to division by zero some of the features became 'infinity'. We simply imputed them with zeros.

## 2. Validation strategy
We used 5-fold GroupKFold split. For each validation set we 10 times randomly chose an assesment for each user resulting in 50 validation sets. The final CV score was computed using their mean. We noticed a strong correlation with public leaderboard.

## 3. Modeling
A 5-fold stack of 5 LGBM models with Ridge regression on top of them. All LGBM models were trained using early stopping by QWK. Here is a brief summary of our first level models:
1. All features. LGBM encoder for assessment title, assessment world and assessment title lag.
2. Subset of top important features from model 1. LGBM encoder for assessment title. 
3. Another subset of top important features from model 1. LGBM encoder for previous assessment titles mode.
4. Model 1 with a different seed.
5. Model 3 with a different seed.

## 4. QWK thresholding
As opposed to most of the teams, we didn't spend any time on optimizing the threshold to convert raw regression predictions to integers. Instead, we found [a paper (in russian)](https://cyberleninka.ru/article/n/o-maksimizatsii-kriteriya-kvadratichnogo-vzveshennogo-kappa/viewer) in which the author mathematically proves that QWK can be directly optimized by adjusting the raw regression predictions to the original train distribution. We compute the adjusted predictions as follows:
```
y_pred =  y_train.mean() + (y_pred_raw - y_pred_raw.mean()) / (y_pred_raw.std() / y_train.std())
```
After that, the predictions were simply rounded to the nearest integer. We didn't notice any significant difference in accuracy between this approach and threshold optimization. However, it helped us to spend less time on thresholding (which may result in overfitting as well) and focus on other things like feature engineering or modeling. In addition, this approach is more desirable since we can directly calculate our validation QWK (and use it for early stopping, for example) without using the true labels.

## 5. How to run the code
Make sure you've downloaded the data and put it in 'data' folder.
```
python preprocessing.py train
python preprocessing.py test_for_train
python preprocessing.py test
python train.py
python predict.py
```
