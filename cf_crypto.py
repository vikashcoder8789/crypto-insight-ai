"""Code for the computation of f1 score, accuracy confusion matrix (step 6 and 7)"""

#importing modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#importing dataset
data = pd.read_csv('emails.csv')
df = data.drop(['Email No.'], axis=1)

#asssigning X and Y
X = df.drop('Prediction', axis=1).values
Y = df['Prediction'].values

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#model selection 
model = SVC(kernel='rbf', gamma='auto')
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

#confusion matrix
cf_matrix = confusion_matrix(Y_test, Y_pred)
tn, fp, fn, tp = cf_matrix.ravel()

accuracy = tp+tn/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*precision*recall/(precision+recall)
print("Accuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1 Score: ",f1_score)
print(tn,fp,fn,tp)

#graphical representation 
sns.heatmap(cf_matrix, annot=True)
plt.show()