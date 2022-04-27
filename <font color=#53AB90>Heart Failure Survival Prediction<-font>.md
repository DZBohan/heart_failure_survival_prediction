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

## <font color=#FFA689>4. Feature Selection</font>

There are 7 continuous features and 5 categorical features in this dataset. First, let's focus on the feature of time. This feature is patients' follow-up period. The entire data collection lasted for more than 9 months. Let's first look at the distribution of this feature.

```
color=["#8cc7b5","#ffc7b5"]
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_copy['time'],bins=15,data=df_copy,hue="Target",palette=color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Fellow-up Period (days)')
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature1.png?raw=true"/></div>

The follow up period for patients ranges from 4 to 284 days. Many of the patients with very short follow-up periods had died early in the data collection period and were therefore lost to contact. We can see high intensity of mortality in the initial days.

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_copy['Target'],y=df_copy['time'],palette = color)
plt.ylabel('Fellow-up Period (days)')
plt.xlabel('Death Events')
```
<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature2.png?raw=true"/></div>

We can also see from the boxplot that the follow-up period was shorter for patients who died. Although this feature has a high correlation with death in heart failure, it is not meaningful for prediction, so I removed it from the dataset.

```
df_copy = df_copy.drop(['time'],axis=1)
df_copy.head(10)
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature3.png?raw=true"/></div>

Now, I have 6 continuous features and 5 categorical features in this dataset. I am going to analyze the continuous features and categorical features separately.

### <font color=#FFA689>4.1 Continous Features</font>

First, I'll analyze and select the continuous features. I remove the categorical features from the dataset and create a new dataframe df_continuous.

```
df_continuous = df_copy.drop(['ANA','DIA','HBP','SEX','SMO'],axis=1)
df_continuous.head(10)
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature4.png?raw=true"/></div>

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
<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature5.png?raw=true"/></div>

As seen in the heatmap, the correlation between continuous features is relatively low. Four features, Serum creatinine, Ejection fraction, Age, and Serum sodium, have relatively high correlations with Target. Two features, CPK and Platelets, had low correlations with Target.

To verify whether the features CKP and Platelets should be removed, I first trained some common models using four features, Serum creatinine, Ejection fraction, Age, and Serum sodium, followed by six features Serum creatinine, Ejection fraction, Age, Serum sodium, CPK and Platelets to train these models. The scores of the two training sessions were compared to determine whether to remove the two features that were less relevant to the Target. Here I use the cross-validation method. The code will be given in Section 5 of this paper.

The table below lists the scores of the two model trainings, and here I used five models for validation, which are logistic regression, decision tree, random forest, XGBoost and GradientBoost.


<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature6.png?raw=true"/></div>

It can be seen that the scores of all models are lower when trained with six features than when trained with four, so I choose to remove these two features and keep only the four continuous features of Serum creatinine, Ejection fraction, Age, and Serum sodium.

```
df_continuous1 = df_continuous.drop(['PLA','CPK'],axis=1)
```

I further used pairplot to see the correlation between these four features.

```
sns.pairplot(df_continuous1,hue="Target",palette=color);
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature7.png?raw=true"/></div>

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

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature8.png?raw=true"/></div>

The normal range of serum creatinine is 0.6 - 1.3 mg/dL. In the histogram, some data points have values higher than 4. I had regarded them as outliers. However, I found that the highest record of serum creatinine is 73.8 mg/dL [3], meaning they are not outliers. 

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['SCR'],palette = color)
plt.ylabel('Serum Creatinine (mg/dL)')
plt.xlabel('Death Events')
ax.set_ylim([0,4])
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature9.png?raw=true"/></div>

As seen in the boxplot, patients are more likely to die in heart failure when their serum creatinine is higher than the normal range.

#### <font color=#FFA689>Ejection Fraction</font>

Let's have a look at the distribution of the feature ejection fraction. 

```
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_continuous1['EFR'],bins=15,data=df_continuous, hue ="Target",palette = color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Ejection Fraction (%)')
```
<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature10.png?raw=true"/></div>

The normal range of the ejection fraction is 50% - 75%. It can be seen in the histogram that more patients had ejection fraction below the normal range.


```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['EFR'],palette = color)
plt.ylabel('Ejection Fraction (%)')
plt.xlabel('Death Events')
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature11.png?raw=true"/></div>

As seen in the boxplot, patients are more likely to die in heart failure when their serum creatinine is lower.

#### <font color=#FFA689>Age</font>

Let's have a look at the distribution of the third feature, age. 

```
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_continuous1['AGE'],bins=15,data=df_continuous1, hue ="Target",palette = color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Age (years)')
```
<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature12.png?raw=true"/></div>

All patients were older than 40 years of age.

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['AGE'],palette = color)
plt.ylabel('Age (years)')
plt.xlabel('Death Events')
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature13.png?raw=true"/></div>

Patients who are older are more likely to die.

#### <font color=#FFA689>Serum Sodium</font>

Let's have a look at the distribution of the last feature, serum sodium. 

```
plt.figure(figsize=(10,7))
ax = sns.histplot(x=df_continuous1['SSO'],bins=15,data=df_continuous1, hue ="Target",palette = color,multiple="stack")
plt.legend(labels=["Died","Survived"],fontsize = 'large')
plt.xlabel('Serum Sodium (mEq/L)')
```
<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature14.png?raw=true"/></div>

The normal range of serum sodiun is 135 - 145 mEq/L. More patients in this dataset had serum sodium in the normal range.

```
plt.figure(figsize=(10,7))
ax = sns.boxenplot(x=df_continuous1['Target'],y=df_continuous1['SSO'],palette = color)
plt.ylabel('Serum Sodium (mEq/L)')
plt.xlabel('Death Events')
ax.set_ylim([125,150])
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature15.png?raw=true"/></div>

As seen in the boxplot, patients are more likely to die in heart failure when their serum sodium is lower than the normal range.


### <font color=#FFA689>4.2 Categorical Features</font>

I remove the continuous features from the df_copy dataframe, keeping only the 5 categorical features, and create a new dataframe df_categorical.

```
df_categorical = df_copy.drop(["AGE","CPK","EFR","PLA","SCR","SSO"],axis=1)
df_categorical.head(10)
```

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature16.png?raw=true"/></div>

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

I used this `cramers_corrected_stat` function to measure the correlation between categorical features in the dataset and to plot the heat map.

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

<div align=center><img width =60% src ="https://github.com/DZBohan/heart_failure_survival_prediction/blob/main/images/feature17.png?raw=true"/></div>

In the categorical features, the correlation between diabetes, sex and smoking and death from heart failure was 0, so I removed these three features directly. Platelets and high blood pressure had a relatively low correlation with target. I used the same method as when screening continuous features to decide whether to keep these two features.

I train the models using the four identified continuous features plus these two categorical features and compare the scores with the scores of the model trained with only four continuous features.