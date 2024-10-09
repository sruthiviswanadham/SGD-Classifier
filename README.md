# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
step 1. start
step 2. Import Necessary Libraries and Load Data
step 3. Split Dataset into Training and Testing Sets
step 4. Train the Model Using Stochastic Gradient Descent (SGD)
step 5. Make Predictions and Evaluate Accuracy
step 6. Generate Confusion Matrix
step 7. end
```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Viswanadham Venkata Sai Sruthi
RegisterNumber: 212223100061
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
```
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
```
## Output:
![image](https://github.com/user-attachments/assets/d1e22442-c22e-44fa-839a-2746d2ee1f78)

```
X=df.drop('target',axis=1)
Y=df['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
```
```
sgd_clf.fit(X_train,Y_train)
```
## Output:
![image](https://github.com/user-attachments/assets/a865fcc4-6fa3-447f-a802-4d9ee8413bf5)
```
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(Y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
## Output:
![image](https://github.com/user-attachments/assets/bc2515cd-88eb-438e-8a84-5adc5960e686)

```
cm=confusion_matrix(Y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:
![image](https://github.com/user-attachments/assets/e7d9efe6-4a29-4aa2-b562-858bad18e8a0)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
