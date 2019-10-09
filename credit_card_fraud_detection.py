import pandas as pd, numpy as np
import keras

np.random.seed(2)

df = pd.read_csv('creditcard.csv')

# Pre-Processing
from sklearn.preprocessing import StandardScaler

df['normalized_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
data = df.drop(['Amount', 'Time'], axis=1)

X = data.iloc[:, data.columns!='Class']
y = data.iloc[:, data.columns=='Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create Neural Net
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential([
    Dense(units=16, input_dim=29, activation='relu'),
    Dense(units=24, input_dim=29, activation='relu'),
    Dropout(0.5),
    Dense(units=20, input_dim=29, activation='relu'),
    Dense(units=24, activation='relu'),
    Dense(units=1, activation='sigmoid'),
])

print(model.summary())

# Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=15, epochs=5)

# Test model
score = model.evaluate(X_test, y_test)
print(score)

from sklearn.metrics import confusion_matrix
y_predict = model.predict(X_test)
y_test = pd.DataFrame(y_test)

cnf_matrix = confusion_matrix(y_test, y_predict.round())
print(cnf_matrix)

