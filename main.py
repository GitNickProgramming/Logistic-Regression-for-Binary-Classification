"""
    References:
        - sciKit learn website documentation on Logistic Regression
        - Class slides
        - Class provide demos
        - Office Hours with Professor
    Description:
        Logistic Regression for Binary Classification using Pima Indians Diabetes Dataset
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 1. & 2. Select and Load Pima Indians Diabetes Data into a pandas Dataframe
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'Outcome']
pima = pd.read_csv('pima-indians-diabetes.csv', header=None, names=col_names)

# 3. Select 5 features and true outcome
feature_cols = ['pregnant', 'skin', 'bmi', 'age', 'glucose']
X = pima[feature_cols]
y = pima['Outcome']

# 4. Split into test and training data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# 5. fit the model with data and test
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# 6. Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ', cnf_matrix)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual Outcome')
plt.xlabel('Predicted Outcome')
plt.figure()    # plot Confusion Matrix


# 6. Precision Score, Recall Score, F Score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# 7. Plot ROC curve and print out the ROC_AUC Score
y_pred_proba = logreg.predict_proba(x_test)[::, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)      # print out AUC score
plt.plot(fpr, tpr, label="Logistic Regression, auc=" + str(auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression for Binary Classification')
plt.legend(loc=4)
plt.show()    # plot ROC Curve
