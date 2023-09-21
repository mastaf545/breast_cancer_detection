># breast_cancer_detection
># PYTHON-Analyzing and MACHINE-BREAST CANCER DETECTION
## What makes name timeless or trendy?
Using data published by the US Social Security Administration, which extends over a period of **one hundred years** to determine whether a resident has cancer or not.

## Creating and importing data 
![](1.png)
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.r
ead_csv('data.csv')
df.head(5)
df['diagnosis'].value_counts()
df['diagnosis']=pd.get_dummies(df['diagnosis'],drop_first=True)
df['diagnosis'].value_counts()

```
-----------------

------------------------

-----------------------------------
## 2 :logisticregression model
```
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression().fit(x_train,y_train)
logmodel.score(x_train,y_train)
logmodel.score(x_test,y_test)
y_pred=logmodel.predict(x_test)
y_pred[:5]
```
------------------------
![](logisticregression.png)
------------------------
## 3 :KNeighborsClassifier model
```
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
```
---------------
![](kneighborsclassifier.png )

------------------------
>## RandomForestClassifier model
```
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,random_state=33)
rfc.fit(x_train,y_train)

rfc.score(x_train,y_train)
rfc.score(x_test,y_test)
y_pred3=rfc.predict(x_test)
y_pred3[:10]

```
------------------------
![](randomforestcalssifier.png )
---------------------------
## decisiontreeclassifier model
```
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_depth=3,random_state=44)
dtc.fit(x_train,y_train)
print(dtc.score(x_train,y_train))
print(dtc.score(x_test,y_test))
y_pred4=dtc.predict(x_test)
y_pred4[:15]
```
----------------------
![]( decisiontreeclassifier.png)
--------------------------
## GaussianNB model
```

from sklearn.naive_bayes import GaussianNB
gau=GaussianNB()
gau.fit(x_train,y_train)
print(gau.score(x_train,y_train))
print(gau.score(x_test,y_test))
y_pred5=gau.predict(x_test)
y_pred_proba=gau.predict_proba(x_test)
print(y_pred5[:10])
y_test[:10].values
print(y_pred_proba[:5])

```
------------------

---------------------------

![](GaussianNB.png )

------------------------------------

-------------------------------
### THANK YOU♥
