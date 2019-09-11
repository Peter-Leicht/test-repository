import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# Create dataframe from dataset
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                         columns=np.append(cancer['feature_names'], ['target']))


# Visualize the data

# Create pairplot
sns.pairplot(df_cancer, hue='target', vars=['mean radius', 'mean texture'])
plt.show()

# Count how many cases are malignant v. benign
sns.countplot(df_cancer['target'])
plt.show()

# Create scatter plot
sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df_cancer)
plt.show()

# Create model

# Split data frames for inputs and target variable
X = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']

# Train, Test, Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Normalize data
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train - min_train)/range_train

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

# Train data

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC(gamma='auto')
svc_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svc_model.predict(X_test_scaled)


# Evaluate model
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred))

# Optimize C and gamma parameters to better the model

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(X_train_scaled, y_train)
print(grid.best_params_)

# Evaluate model that uses optimized C and gamma parameters

grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)

sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, grid_predictions))
