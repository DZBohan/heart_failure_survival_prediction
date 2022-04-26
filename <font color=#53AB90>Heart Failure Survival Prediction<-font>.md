# <center> <font color=#53AB90>Heart Failure Survival Prediction</font> </center>

<div align=center><img width =100% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/title.png?raw=true"/></div>

## <font color=#FFA689>1. Introduction</font>

Heart failure is a serious condition in which the heart is unable to produce enough blood to supply the entire body. Approximately 6.2 million adults in the United States have heart failure. In 2018, close to 400,000 deaths across the United States were associated with heart failure [1].

The dataset used in this study was from the UC Irvine Machine Learning Repository and was collected by Krembil Research Institute, Canada. All patients in the dataset were from Pakistan, and data were collected from a specific time of April to December 2015 [2].

Death of heart failure is associated with location [1]. Prior to this dataset, there were few studies related to heart failure in Pakistani, so this dataset is of great importance.

The goal of this study is to select appropriate features and models to predict survival in heart failure based on this dataset. This will enable hospitals to have a clearer picture of the patients’ condition when admitting them and make appropriate preparations and treatment plans.

I will also solve some problem when training models with small size datasets. I used jupter notebook to do this project.

## <font color=#FFA689>2. Importing Libraries</font>

The following libraries were used in this project.


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import itertools
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
```


## <font color=#FFA689>3. Dataset</font>

Let's first look at the information on the dataset provided by the UC Irvine Machine Learning Repository.

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset1.png?raw=true"/></div>

The dataset contains 299 samples and 13 features. The good news is that there are no missing values.

### <font color=#FFA689>3.1 Loading the Dataset</font>

Let's load the dataset and see the description.

```
df = pd.read_csv('../datasets/heart_failure_clinical_records_dataset.csv')
df.describe().T
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset2.png?raw=true"/></div>

I got the following information from the description of the dataset.

* In addition to my target, DEATH_EVENT, there are 12 features in the dataset
* The dataset has two types of features, categorical features, such as sex, and continuous features, such as age
* Checking the relevant information shows that the data points in the dataset are reliable

I created the following table to explain the meaning of each feature and to distinguish between categorical features and continuous features.

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset3.png?raw=true"/></div>

Of the 12 features, 5 are categorical features, and 7 are continuous features. To facilitate the analysis later, I give each feature a symbol, which is noted in the last column of the table.

Now, I replace the feature names in the dataset with symbols.

```
df_copy = df.rename(columns={'age':'AGE','creatinine_phosphokinase':'CPK','ejection_fraction':'EFR','platelets':'PLA','serum_creatinine':'SCR','serum_sodium':'SSO','anaemia':'ANA','diabetes':'DIA','high_blood_pressure':'HBP','sex':'SEX','smoking':'SMO','DEATH_EVENT':'Target'})
df_copy.head(10)
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset4.png?raw=true"/></div>

### <font color=#FFA689>3.2 Evaluating the Target</font>

Now, let's visualize the target, death event of heart failure.

```
color=["#8cc7b5","#ffc7b5"]
plt.figure(figsize=(10,7))
sns.countplot(x= df_copy["Target"], palette= color)
plt.xlabel('Heart Failure Death Event')
plt.ylabel('Count')
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset5.png?raw=true"/></div>

There is an imbalance in the target of the dataset, so I will come to solve this problem later.

### <font color=#FFA689>3.3 Features Distribution</font>

Let's take a look at the distribution of features in general, and later I will analyze specifically the features that will be used in this project.

```
index = 0
plt.figure(figsize=(20, 10))
feature = ["AGE","CPK","EFR","PLA","SCR","SSO","ANA","DIA","HBP","SEX","SMO","time"]
for i in feature:
    index += 1
    plt.subplot(3, 4, index)
    ax = sns.histplot(x=df_copy[i],bins=15,data=df_copy, hue ="Target",palette = color,multiple="stack")
    plt.legend(labels=["Died","Survived"],fontsize = 'large')
    plt.xlabel(i)
plt.show()
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset6.png?raw=true"/></div>

## <font color=#FFA689>4. Feature Analysis</font>

There are 7 continuous features and 5 categorical features in this dataset. First, let's focus on the feature of time. This feature is patients' follow-up period. The entire data collection lasted for more than 9 months. Let's first look at the distribution of this feature.

```
color=["#8cc7b5","#ffc7b5"]
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_copy['time'],bins=15,data=df_copy,hue="Target",palette=color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Fellow-up Period (days)')
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature1.png?raw=true"/></div>

The follow up period for patients ranges from 4 to 284 days. Many of the patients with very short follow-up periods had died early in the data collection period and were therefore lost to contact. Therefore, we can see high intensity of mortality in the initial days.

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_copy['Target'],y=df_copy['time'],palette = color)
plt.ylabel('Fellow-up Period (days)')
plt.xlabel('Death Events')
```
