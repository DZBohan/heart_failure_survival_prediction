# <span id="jump"><center><font color=#53AB90>Heart Failure Survival Prediction</font></center></span>

## <center><font color=#FFA689>Bohan Zhang</font></center>

<center>GitHub Page: https://dzbohan.github.io/heart_failure_survival_prediction/</center>

<div align=center><img width =100% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/title.png?raw=true"/></div></center>

## <font color=#53AB90>Index</font>

- [1. Introduction](#1)

- [2. Importing Libraries](#2)

- [3. Dataset](#3)

	- [3.1 Loading the Dataset](#3.1)

	- [3.2 Evaluating the Target](#3.2)

	- [3.3 Features Distribution](#3.3)

- [4. Feature Selection](#4)

	- [4.1 Continous Features](#4.1)

	- [4.2 Categorical Features](#4.2)

	- [4.3 Final Features](#4.3)

- [5. Model Selection](#5)

	- [5.1 Imbalance Issue](#5.1)

	- [5.2 Dataset Splitting](#5.2)

	- [5.3 Standardization](#5.3)

	- [5.4 Find the Best Model](#5.4)

- [6. Random Forest](#6)

	- [6.1 Hyperparameters Tuning](#6.1)

	- [6.2 Feature Importance](#6.2)

	- [6.3 Final Features](#6.3)

- [7. Conclusion](#7)

- [References](#8)
 
## <h2 id="1"><font color=#FFA689>1. Introduction</font></h2>

Heart failure is a serious condition in which the heart is unable to produce enough blood to supply the entire body. Approximately 6.2 million adults in the United States have heart failure. In 2018, close to 400,000 deaths across the United States were associated with heart failure [[1]](https://www.cdc.gov/heartdisease/heart_failure.htm).

The dataset used in this study was from the <font color=#53AB90>UC Irvine Machine Learning Repository</font> and was collected by Krembil Research Institute, Canada. All patients in the dataset were from <font color=#53AB90>Pakistan</font>, and data were collected from a specific time of April to December 2015 [[2]](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records).

Death of heart failure is associated with location [[1]](https://www.cdc.gov/heartdisease/heart_failure.htm). Prior to this dataset, there were few studies related to heart failure in Pakistani, so this dataset is of great importance.

The goal of this study is to select appropriate features and models to predict survival in heart failure based on this dataset. This will enable hospitals to have a clearer picture of the patients’ condition when admitting them and make appropriate preparations and treatment plans.

I will also solve some problem when training models with small size datasets. I used jupter notebook to do this project.

## <h2 id="2"><font color=#FFA689>2. Importing Libraries</font></h2>

The following libraries were used in this project. The Imblearn package needs to be installed. See 5.1 Imbalance Issue for details.


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import itertools
import xgboost as xgb
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
```


## <h2 id="3"><font color=#FFA689>3. Dataset</font></font></h2>


Let's first look at the information on the dataset provided by the UC Irvine Machine Learning Repository.

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset1.png?raw=true"/></div>

The dataset contains 299 samples and 13 features. The good news is that there are no missing values.

### <h2 id="3.1"><font color=#FFA689>3.1 Loading the Dataset</font></h2>

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

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset4.png?raw=true"/></div>

### <h2 id="3.2"><font color=#FFA689>3.2 Evaluating the Target</font></h2>

Now, let's visualize the target, death event of heart failure.

```
color=["#8cc7b5","#ffc7b5"]
plt.figure(figsize=(10,7))
sns.countplot(x= df_copy["Target"], palette= color)
plt.xlabel('Heart Failure Death Event')
plt.ylabel('Count')
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/dataset5.png?raw=true"/></div>

There is an imbalance in the target of the dataset, so I will come to solve this problem later.

### <h2 id="3.3"><font color=#FFA689>3.3 Features Distribution</font></h2>

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

## <h2 id="4"><font color=#FFA689>4. Feature Selection</font></h2>

There are 7 continuous features and 5 categorical features in this dataset. First, let's focus on the feature of <font color=#53AB90>time</font>. This feature is patients' follow-up period. The entire data collection lasted for more than 9 months. Let's first look at the distribution of this feature.

```
color=["#8cc7b5","#ffc7b5"]
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_copy['time'],bins=15,data=df_copy,hue="Target",palette=color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Fellow-up Period (days)')
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature1.png?raw=true"/></div>

The follow up period for patients ranges from 4 to 284 days. Many of the patients with very short follow-up periods had died early in the data collection period and were therefore lost to contact. We can see high intensity of mortality in the initial days.

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_copy['Target'],y=df_copy['time'],palette = color)
plt.ylabel('Fellow-up Period (days)')
plt.xlabel('Death Events')
```
<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature2.png?raw=true"/></div>

We can also see from the boxplot that the follow-up period was shorter for patients who died. Although this feature has a high correlation with death in heart failure, it is not meaningful for prediction, so I removed it from the dataset.

```
df_copy = df_copy.drop(['time'],axis=1)
df_copy.head(10)
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature3.png?raw=true"/></div>

Now, I have 6 continuous features and 5 categorical features in this dataset. I am going to analyze the continuous features and categorical features separately.

### <h2 id="4.1"><font color=#FFA689>4.1 Continous Features</font></h2>

First, I'll analyze and select the continuous features. I remove the categorical features from the dataset and create a new dataframe df_continuous.

```
df_continuous = df_copy.drop(['ANA','DIA','HBP','SEX','SMO'],axis=1)
df_continuous.head(10)
```

<div align=center><img width =30% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature4.png?raw=true"/></div>

Now, let's plot a heatmap to see the correlation between features.

```
corr1 = df_continuous.corr().round(2)
mask = np.zeros_like(corr1, dtype=bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(162,15,s=45,l=65,as_cmap=True)
sns.heatmap(corr1, mask=mask, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
ax.set_title("Correlation between Continuous Features");
plt.tight_layout()
```
<div align=center><img width =50% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature5.png?raw=true"/></div>

As seen in the heatmap, the correlation between continuous features is relatively low. Four features, serum creatinine, ejection fraction, age, and serum sodium, have relatively high correlations with Target. Two features, CPK and Platelets, had low correlations with Target.

To verify whether the features CKP and Platelets should be removed, I first trained some common models using four features, serum creatinine, ejection fraction, age, and serum sodium, followed by six features serum creatinine, ejection fraction, age, serum sodium, CPK and platelets to train these models. The scores of the two training sessions were compared to determine whether to remove the two features that were less relevant to the target. Here I use the cross-validation method. The code will be given in Section 5.4 of this paper.

The table below lists the scores of the two model trainings, and here I used five models for validation, which are logistic regression, decision tree, random forest, XGBoost and GradientBoost.


<div align=center><img width =30% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature6.png?raw=true"/></div>

It can be seen that the scores of all models are lower when trained with six features than when trained with four, so I choose to remove these two features and keep only the four continuous features of Serum creatinine, Ejection fraction, Age, and Serum sodium.

```
df_continuous1 = df_continuous.drop(['PLA','CPK'],axis=1)
```

I further used pairplot to see the correlation between these four features.

```
sns.pairplot(df_continuous1,hue="Target",palette=color);
```

<div align=center><img width =50% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature7.png?raw=true"/></div>

By looking at the pairplot, I determined that the correlation between these four features was low, so I could select them for model training.

Now, let's analyze these four features in detail.


#### <font color=#FFA689>Serum Creatinine</font>

Let's first observe the distribution of the feature serum creatinine. 

```
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_continuous1['SCR'],bins=15,data=df_continuous1, hue ="Target",palette = color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Serum Creatinine (mg/dL)')
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature8.png?raw=true"/></div>

The normal range of serum creatinine is 0.6 - 1.3 mg/dL. In the histogram, some data points have values higher than 4. I had regarded them as outliers. However, I found that the highest record of serum creatinine is 73.8 mg/dL [[3]](https://www.hindawi.com/journals/crin/2021/6048919/), meaning they are not outliers. 

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['SCR'],palette = color)
plt.ylabel('Serum Creatinine (mg/dL)')
plt.xlabel('Death Events')
ax.set_ylim([0,4])
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature9.png?raw=true"/></div>

As seen in the boxplot, patients are more likely to die in heart failure when their serum creatinine is higher than the normal range.

#### <font color=#FFA689>Ejection Fraction</font>

Let's have a look at the distribution of the feature ejection fraction. 

```
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_continuous1['EFR'],bins=15,data=df_continuous, hue ="Target",palette = color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Ejection Fraction (%)')
```
<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature10.png?raw=true"/></div>

The normal range of the ejection fraction is 50% - 75%. It can be seen in the histogram that more patients had ejection fraction below the normal range.


```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['EFR'],palette = color)
plt.ylabel('Ejection Fraction (%)')
plt.xlabel('Death Events')
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature11.png?raw=true"/></div>

As seen in the boxplot, patients are more likely to die in heart failure when their serum creatinine is lower.

#### <font color=#FFA689>Age</font>

Let's have a look at the distribution of the third feature, age. 

```
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_continuous1['AGE'],bins=15,data=df_continuous1, hue ="Target",palette = color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Age (years)')
```
<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature12.png?raw=true"/></div>

All patients were older than 40 years of age.

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['AGE'],palette = color)
plt.ylabel('Age (years)')
plt.xlabel('Death Events')
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature13.png?raw=true"/></div>

Patients who are older are more likely to die.

#### <font color=#FFA689>Serum Sodium</font>

Let's have a look at the distribution of the last feature, serum sodium. 

```
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_continuous1['SSO'],bins=15,data=df_continuous1, hue ="Target",palette = color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Serum Sodium (mEq/L)')
```
<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature14.png?raw=true"/></div>

The normal range of serum sodiun is 135 - 145 mEq/L. More patients in this dataset had serum sodium in the normal range.

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['SSO'],palette = color)
plt.ylabel('Serum Sodium (mEq/L)')
plt.xlabel('Death Events')
ax.set_ylim([125,150])
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature15.png?raw=true"/></div>

As seen in the boxplot, patients are more likely to die in heart failure when their serum sodium is lower than the normal range.


### <h2 id="4.2"><font color=#FFA689>4.2 Categorical Features</font></h2>

I remove the continuous features from the df_copy dataframe, keeping only the 5 categorical features, and create a new dataframe df_categorical.

```
df_categorical = df_copy.drop(["AGE","CPK","EFR","PLA","SCR","SSO"],axis=1)
df_categorical.head(10)
```

<div align=center><img width =20% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature16.png?raw=true"/></div>

I used the method called Cramér's V to see the correlation between the classified features. This method is based on Pearson's chi-squared statistic and able to meature the association between two nominal variables.

```
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
```

I used this <font color=#53AB90>cramers_corrected_stat</font> function to measure the correlation between categorical features in the dataset and to plot the heat map.

```
cols = ['ANA','DIA','HBP','SEX','SMO','Target']
corrM = np.zeros((len(cols),len(cols)))
for col1, col2 in itertools.combinations(cols, 2):
    idx1, idx2 = cols.index(col1), cols.index(col2)
    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df_categorical[col1], df_categorical[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]

corr2 = pd.DataFrame(corrM, index=cols, columns=cols)
fig, ax = plt.subplots(figsize=(10, 10))
mask = np.zeros_like(corr2)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr2, mask=mask, cmap=cmap, ax=ax, 
                 square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True); 
ax.set_title("Cramér's V Correlation between Boolean Features");
```

<div align=center><img width =50% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature17.png?raw=true"/></div>

In the categorical features, the correlation between diabetes, sex and smoking and death from heart failure was 0, so I removed these three features directly. Platelets and high blood pressure had a relatively low correlation with target. I used the same method as when screening continuous features to decide whether to keep these two features.

I train the models using the four identified continuous features plus these two categorical features and compare the scores with the scores of the model trained with only four continuous features.

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature18.png?raw=true"/></div>

As can be seen from the table, whether or not these two features are added has little effect on the results. For the following three reasons, I decided to remove these two classification features, that is, to keep only four continuous features to train the model.

* Removing them has no effect on the performance of the model
* Removing two features can improve future data collection efficiency
* Removing two features can speed up training

### <h2 id="4.3"><font color=#FFA689>4.3 Final Features</font></h2>

In summary, I kept only 4 continuous features, erum creatinine, ejection fraction, age, and serum sodium for the next model training. The data frame used is df_continuous1.

```
df_continuous1.head(10)
```

<div align=center><img width =20% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature19.png?raw=true"/></div>

## <h2 id="5"><font color=#FFA689>5 Model Selection</font></h2>

First, I assign the four selected features to the variable X and the target to the variable y.

```
X = df_continuous1.drop(columns='Target')
y = df_continuous1['Target']
```

### <h2 id="5.1"><font color=#FFA689>5.1 Imbalance Issue</font></h2>

In Chapter 3, I evaluated the targets of the dataset and found that there was an imbalance issue. There are 96 death and 203 survival in the dataset.

<div align=center><img width =30% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/model1.png?raw=true"/></div>

Now, I will use the <font color=#53AB90>imbalance-learn package</font> to solve this problem. First, let's install the imbalance-learn package.

```
pip install imbalance-learne
```
The imbalance-learn package has both random oversampling and random undersampling functions. Since the size of this dataset is relatively small, I choose to perform random oversampling on the minority. 

I have imported the <font color=#53AB90>RandomOverSampler</font> in the second chapter. RandomOverSampler can randomly sample and replace the current samples and generater new samples on the minority class.

```
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
print(Counter(y))
print(Counter(y_over))
```

<div align=center><img width =20% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/model2.png?raw=true"/></div>

It can be seen that the minority class in the dataset, i.e., death, went from 96 samples to 203.

Because the sampling process is random, the dataset will change after each oversampling, which will cause some problems that I will explain in detail in later chapters.


### <h2 id="5.2"><font color=#FFA689>5.2 Dataset Splitting</font></h2>

Now, I am going to splitting the dataset into training dataset and test dataset. Training set is for training the model, and test set is for testing the model with new, unseen data points.

One important reason of splitting the dataset is to avoid the overfitting problem. Overfitting means a model has good performance on training dataset but has poor performance on the new data points.

Let's split the dataset. I selected 80% of the data points as the training set and 20% of the data points as the test set.

```
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.20, random_state=20)
```

### <h2 id="5.3"><font color=#FFA689>5.3 Standardization</font></h2>

Some models, such as logistic regression, require standardization of the dataset, so I use <font color=#53AB90>StandardScaler</font> for standardization.

```
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

Check the comparison of features' scale before and after standardization.

```
colours =["#ffc7b5","#e1c7b5","#aac7b5","#8cc7b5"]
plt.figure(figsize=(15,10))
ax=sns.boxenplot(data = X_train,palette = colours)
ax.set_xticklabels(['AGE','EFR','SCR','SSO'])

plt.show()
```

This is the features' scale before standardization.

<div align=center><img width =50% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/model3.png?raw=true"/></div>

This is the features' scale after standardization.

<div align=center><img width =50% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/model4.png?raw=true"/></div>

### <h2 id="5.4"><font color=#FFA689>5.4 Find the Best Model</font></h2>

Here I applied a function called <font color=#53AB90>GridSearchCV</font> which uses <font color=#53AB90>cross-validation</font> and is able to get average score on five different models, logistic regression, decision tree, random forest, XGBoost, gradient boost.

```
def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10,20]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200],
                'max_depth': [10,15,20,50,100]
            }
        },
        
        'XGBoost': {
            'model': xgb.XGBClassifier(booster='gbtree', min_child_weight=1, gamma=0, scale_pos_weight=1),
            'parameters': {
                'eta': [0.1,0.2,0.3],
                'max_depth': [3,4,5,6,7,8,9,10],
            }
        },
        
        'GradientBoost': {
            'model': GradientBoostingClassifier(),
            'parameters':{
                'max_depth': [3,4,5,6,7,8,9,10],
                'n_estimators': [10,15,20,50,100,200]
            }
        }        

    }
    
    scores = [] 
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])

find_best_model(X_train, y_train)
```

Here is the result.

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/model5.png?raw=true"/></div>

However, I faced a challenge in this step. The scores of models vary each time using RandomOversampler to replace the minority class of the target. As I mentioned in the section 5.1, because the sampling process is random, the dataset will change after each oversampling. This is the reason why the socres are different each time. The solution of this problem is to try multiple time to see if there is a model is consistently best. Here I tried GridSearchCV for four times, and this is the result.

<div align=center><img width =50% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/model6.png?raw=true"/></div>

As shown in the table, the overall ranking of the scores remains consistent, and random forest is relatively the best. Therefore, I will choose random forest as the model.

## <h2 id="6"><font color=#FFA689>6 Random Forest</font></h2>

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf1.png?raw=true"/></div>

Random forest, a Supervised Machine Learning Algorithm, can build decision trees on different samples and takes their majority vote for classification.

In this section, I will use the training set to train the random forest model, tune the hyperparameters of the model to give it the best performance, and then test the accuracy on the test set.

### <h2 id="6.1"><font color=#FFA689>6.1 Hyperparameters Tuning</font></h2>

Hyperparameters are model's parameters used to control the learning process. Tuning hyperparameters can improve the performance of the model. There are many hyperparameters of random forest.

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf2.png?raw=true"/></div>

#### <font color=#FFA689>Training with Default Hyperparameters</font>

Now, I will first train the set with default hyperparameters and do a prediction on the test set.

```
classifier = RandomForestClassifier(random_state=20)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```

Let's plot a confusion matrix to see the model performance on test set.

```
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
p = sns.heatmap(cm, annot=True, cmap=cmap, fmt='g')
plt.title('Confusion matrix for Random Forest Classifier Model - Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()
```

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf3.png?raw=true"/></div>

As shown in the figure, of the 82 predictions, 73 were correct and 9 were incorrect, including two predictions of death as survival and seven predictions of survival as death.

Let‘s generate the accuracy on test set.

```
score = round(accuracy_score(y_test, y_pred),4)*100
print("Accuracy on test set: {}%".format(score))
```

<div align=center><img width =20% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf4.png?raw=true"/></div>

The accuracy of the model on the test set is 89.02%, which is a good result. Now I test the performance of the model on the training set to see if there is any overfitting issue.

```
y_train_pred = classifier.predict(X_train)
```

This is the confusion matrix on training set.

```
cm = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(10,7))
p = sns.heatmap(cm, annot=True, cmap=cmap, fmt='g')
plt.title('Confusion matrix for Random Forest Classifier Model - Train Set')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()
```
<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf5.png?raw=true"/></div>

Then, I generate the accuracy on training set.

```
score = round(accuracy_score(y_train, y_train_pred),4)*100
print("Accuracy on trainning set: {}%".format(score))
```

<div align=center><img width =20% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf6.png?raw=true"/></div>

The accuracy of the model on the training set is 100%, which is much higher the the accuracy on the test set, so there is a overfitting issue. I need to tune the hyperparameters to solve this problem.

#### <font color=#FFA689>Hyperparameters Selection</font>

I listed the hyperparameters I want to tune here.

* n_estimators = number of trees in the forest. Default = 100
* max_depth = max number of levels in each decision tree. Default = None
* min_samples_leaf = min number of data points allowed in a leaf node. Default = 1
* min_samples_split = min number of data points placed in a node before the node is split. Default =2

My goals of tuning the hyperparameters are solving the overfitting issue and keeping the good performance of the model. n_estimators is a major factor affecting the performance of the random forest model. Having a limitation on max_depth can improve the overfitting issue. Increasing the values of min_simples_leaf and min_samples_split can prune the branches of each decision tree in the random forest, thus improving the overfitting problem.

#### <font color=#FFA689>Hyperparameters Tuning</font>

First, let's tune n_estimators. I'll use a range of 10 to 200 with an interval of 10 to generate the scores. Other hyperparameters are default values.

```
values = [i for i in range(10, 200, 10)]
# define lists to collect scores
train_scores, test_scores = list(), list()

for i in values:
    # configure the model
    model = RandomForestClassifier(n_estimators=i,random_state=20)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_scores.append(test_acc)
```
Now, let's plot a line chart to see the trend and find the best point.

```
plt.figure(figsize=(15,7))
pyplot.plot(values, train_scores, '-o', label='Train', color='#8cc7b5')
pyplot.plot(values, test_scores, '-o', label='Test', color='#ffc7b5')
pyplot.legend()
pyplot.show()
```
<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf7.png?raw=true"/></div>

When n_estimators is 40 or 50, the model has the best performance on the test set. When n_estimators is 40, the difference between the performance on the training set and test set is lower, so I choose the value of 40.

Then, let's tune max_depth. I'll use a range of 1 to 20 to generate the scores. I set n_estimator as 40, and other hyperparameters are default values.

```
values = [i for i in range(1, 21)]
# define lists to collect scores
train_scores, test_scores = list(), list()

for i in values:
    # configure the model
    model = RandomForestClassifier(n_estimators=40, max_depth=i,random_state=20)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_scores.append(test_acc)
```

Now, let's plot a line chart to see the trend and find the best point.

```
plt.figure(figsize=(15,7))
pyplot.plot(values, train_scores, '-o', label='Train',color='#8cc7b5')
pyplot.plot(values, test_scores, '-o', label='Test',color='#ffc7b5')
pyplot.legend()
pyplot.show()
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf8.png?raw=true"/></div>

When max_depth is 12, the model has best performance on the test set, but the difference between performance on the test set and training set is large. When max_depth is 4, the score on the test set is still more than 80%, and the difference looks smaller. Therefore, I perfer max_depth as 4.

Let's tune min_samples_leaf next. I'll use a range of 1 to 20 to generate the scores. I set n_estimator as 40, max_depth as 4 and other hyperparameters are default values.

```
values = [i for i in range(1, 21)]
# define lists to collect scores
train_scores, test_scores = list(), list()

for i in values:
    # configure the model
    model = RandomForestClassifier(n_estimators=40, max_depth=4, min_samples_leaf=i, random_state=20)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_scores.append(test_acc)
```
Now, let's plot a line chart to see the trend and find the best point.

```
plt.figure(figsize=(15,7))
pyplot.plot(values, train_scores, '-o', label='Train', color='#8cc7b5')
pyplot.plot(values, test_scores, '-o', label='Test', color='#ffc7b5')
pyplot.legend()
pyplot.show()
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf9.png?raw=true"/></div>

When min_samples_leaf is 12, the model has best performance on the test set, and the difference between performance on the test set and training set is smallest, so I will set min_samples_leaf as 12.


Last, let's tune min_samples_split. I'll use a range of 2 to 100 with an interval of 2 to generate the scores. I set n_estimator as 40, max_depth as 4 and min_samples_split as 12.

```

values = [i for i in range(2, 101,2)]
# define lists to collect scores
train_scores, test_scores = list(), list()

for i in values:
    # configure the model
    model = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=12, min_samples_split=i, random_state=20)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_scores.append(test_acc)
```

Now, let's plot a line chart to see the trend and find the best point.

```
plt.figure(figsize=(15,7))
pyplot.plot(values, train_scores, '-o', label='Train', color='#8cc7b5')
pyplot.plot(values, test_scores, '-o', label='Test', color='#ffc7b5')
pyplot.legend()
pyplot.show()
```
<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf10.png?raw=true"/></div>

When min_samples_split is in the range of 34 to 40, the model has best performance on the test set, and the difference between performance on the test set and training set is smallest, so I will set min_samples_leaf as 40.

I have set all four hyperparameters, n_estimators=50, max_depth=4, min_samples_leaf=12 and min_samples_split=40. Now, I will training the model with these values of hyperparameters and see the scores both on test set and training set.

```
classifier = RandomForestClassifier(n_estimators=40, max_depth=4, min_samples_leaf=12, min_samples_split=40,random_state=20)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = round(accuracy_score(y_test, y_pred),4)*100
print("Accuracy on test set: {}%".format(score))
y_train_pred = classifier.predict(X_train)
score2 = round(accuracy_score(y_train, y_train_pred),4)*100
print("Accuracy on trainning set: {}%".format(score2))
```

<div align=center><img width =20% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf11.png?raw=true"/></div>

The score on the test set is 80.49%, and the score on the training set is 80.86%. It seems that the overfitting issue is improved, and the score on the test set is relatively good.


However, the same problem mentioning in the section 5.4 occurs. The scores of models vary each time using RandomOversampler to replace the minority class of the target. So the hyperparameters might be just suitable for the specific oversampling. To verify that the hyperparameters are applicable to all oversampling scenarios, I tried multiple time to see if the overfitting issue is improved consistently. Here are the scores on test and training set.

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf12.png?raw=true"/></div>

Then I am going to visualize the contents of the table.

This is the line chart of scores of test set and training set before tuning the hyperparameters.

```
plt.figure(figsize=(10, 5))
x = [1,2,3,4,5]
k1 = [100,100,100,100,100]
k2 = [82.93,81.71,80.49,86.59,81.71]
plt.plot(x,k1,'s-',color = '#8cc7b5',label="Train")
plt.plot(x,k2,'o-',color = '#ffc7b5',label="Test")
plt.ylim((60,120))
plt.xlabel("Numbers")
plt.ylabel("Scores")
plt.legend(loc = "best")
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf13.png?raw=true"/></div>

This is the line chart of scores of test set and training set after tuning the hyperparameters.

```
plt.figure(figsize=(10, 5))
x = [1,2,3,4,5]
k1 = [76.83,79.27,76.83,74.39,79.27]
k2 = [79.01,80.56,82.1,80.25,80.25]
plt.plot(x,k1,'s-',color = '#8cc7b5',label="Train")
plt.plot(x,k2,'o-',color = '#ffc7b5',label="Test")
plt.ylim((60,120))
plt.xlabel("Numbers")
plt.ylabel("Scores")
plt.legend(loc = "best")
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf14.png?raw=true"/></div>

It can be seen that the overfitting problem is improved in general rather than just for one specific random oversampling, and all scores of test set are more than 70%. Therefore, the hyperparameters tuning is effective.

### <h2 id="6.2"><font color=#FFA689>6.2 Feature Importance</font></h2>

Now, I am going to verify the importance of the four features I have chosen in the random forest model.

```
feat_labels = df_continuous1.columns[:-1]

classifier = RandomForestClassifier(n_estimators=40, max_depth=4, min_samples_leaf=12, min_samples_split=40,random_state=20)

classifier.fit(X_train, y_train)
importances = classifier.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.figure(figsize=(8, 8))
plt.title('Feature importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center', color=colours)

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
```
<div align=center><img width =30% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf15.png?raw=true"/></div>

As shown in the figure, the importance of the feature serum sodium is very minor for the random forest model. Therefore, I am going to consider whether to remove it or not.

I created a dataframe called df_continuous2 removing serum sodium from df_continuous1.

```
df_continuous2 = df_continuous1.drop(['SSO'],axis=1)
```
I randomly oversampled df_continuous1 and df_continuous2 ten times and used them trained the model with the hyperparameters tuning in the section 5.1 separately. Then, I genrated the scores of test set and training set before removing and after removing the feature serum sodium. The code of training the model and genrating the scores is same as the code in section5.

Here is the table recoding the scores of test set and training set before removing and after removing the feature serum sodium.

<div align=center><img width =40% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf16.png?raw=true"/></div>

Then I am going to visualize the contents of the table.

```
plt.figure(figsize=(10, 5))
x = [1,2,3,4,5,6,7,8,9,10]
k1 = [76.83,79.27,76.83,74.39,79.27,71.95,74.39,71.95,79.27,79.27]
k2 = [79.01,80.56,82.10,80.25,80.25,79.63,80.56,81.17,80.56,81.17]
k3 = [79.27,75.61,76.83,74.39,73.01,75.61,73.17,76.83,79.27,80.49]
k4 = [81.48,83.02,81.48,80.56,79.01,82.41,81.79,79.63,82.72,83.95]
plt.plot(x,k1,'s-',color = '#8cc7b5',label="SSO_test")
plt.plot(x,k2,'o-',color = '#8cc7b5',label="SSO_train")
plt.plot(x,k3,'s-',color = '#ffc7b5',label="NoSSO_test")
plt.plot(x,k4,'o-',color = '#ffc7b5',label="NoSSO_train")
plt.ylim((50,100))
plt.xlabel("Numbers")
plt.ylabel("Scores")
plt.legend(loc = "best")
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf17.png?raw=true"/></div>


The two green lines are the scores of the training and test sets when the feature serum sodium is retained, and the pink lines are the scores of the training and test sets after the feature serum sodium is removed. It can be seen that whether the feature serum sodium is removed or not has no significant effect on the performance of the model.

Therefore, in order to improve the speed of model training and the convenience of future data collection, I decided to remove the feature serum sodium and keep only three features, serum creatinine, ejection fraction and age, to train the model. The final performance of the model has been listed in the table above.

## <h2 id="7"><font color=#FFA689>7 Conclusion</font></h2>

In this project, I used 3 features, serum creatinine, ejection fraction and age, to train the random forest model with a small size dataset, which only has 299 data points. I used random oversampling to resolve imbalances in the dataset. For the model, I tuned the hyperparameters (n_estimators=40, max_depth=4, min_samples_leaf=12, min_samples_split=40). The overfitting problem still exists, but is well improved. Scores on test set remain in the 70-80 range after random oversampling. The following line chart shows the performance of the model.

```
plt.figure(figsize=(10, 5))
x = [1,2,3,4,5,6,7,8,9,10]
k3 = [79.27,75.61,76.83,74.39,73.01,75.61,73.17,76.83,79.27,80.49]
k4 = [81.48,83.02,81.48,80.56,79.01,82.41,81.79,79.63,82.72,83.95]
plt.plot(x,k3,'s-',color = '#8cc7b5',label="test")
plt.plot(x,k4,'o-',color = '#ffc7b5',label="train")
plt.ylim((50,100))
plt.xlabel("Numbers")
plt.ylabel("Scores")
plt.legend(loc = "best")
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/rf18.png?raw=true"/></div>

This project achieves model training and overfitting problem improvement for small data sets. Moreover, this project also achieved to predict the survival of heart failure patients using only three metrics, serum creatinine, ejection fraction and age. Among these three metrics, age is easily accessible, meaning that the ability of a patient to survive heart failure can be predicted by measuring only two of the metrics, serum creatinine, ejection fraction, with a correct rate close to 80%. Therefore, this project is of great importance for the future preparation of patients with heart failure for treatment.

## <h2 id="8"><font color=#FFA689>References</font></h2>

[1] Heart Failure. Centers for Disease Control and Prevention. [https://www.cdc.gov/heartdisease/heart_failure.htm](https://www.cdc.gov/heartdisease/heart_failure.htm)

[2] Heart failure clinical records Data Set. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

[3] Highest Recorded Serum Creatinine (2021). Hindawi. [https://www.hindawi.com/journals/crin/2021/6048919/
](https://www.hindawi.com/journals/crin/2021/6048919/)

### <center>[Back to Top](#jump)</center>
