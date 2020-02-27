
# Select a dataset with binary target values.
from sklearn.datasets import load_diabetes


# Use pandas to read CSV file as dataframe.
import pandas as panda

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = panda.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)
print(pima)

# select 5 features 
feature_cols = ['pregnant', 'skin', 'bmi', "pedigree", 'label']
X = pima[feature_cols]
print(X)

# Use "train_test_split" from "sklearn.cross_validationtrain" to split test/training data
from sklearn.model_selection import train_test_split

pimaSK = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(pimaSK.data, pimaSK.target, test_size=0.4, random_state=0)


# Fit your model with training data and test your model after fitting.
from sklearn.linear_model import LogisticRegression 
logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test[0].reshape(1, -1))
logisticRegr.predict(x_test[0:10])
predictions = logisticRegr.predict(x_test)


# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)

# Calculate and Plot out confusion matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# Seaborn
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

# Matplotlib
import numpy as np
plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shapefor 

for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')

  