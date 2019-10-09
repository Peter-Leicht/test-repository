import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import random

# The 10 classes (or labels) are defined as follows:
# 0 => T-shirt / top
# 1 => Pants
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle Boot

# Create dataframe from dataset
fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep=",")
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep=",")

# Create array to visualize image
training = np.array(fashion_train_df, dtype='float32')
testing = np.array(fashion_test_df, dtype='float32')

# Visualize a random image
i = random.randint(1, 60000)

label = training[i, 0]
print(label)

plt.imshow(training[i, 1:].reshape(28, 28))
plt.show()

# View images in grid format

# Define grid dimensions
W_grid = 15
L_grid = 15

# Return figure object and axes object
# Use axes object to plot specific figures at various locations
fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
axes = axes.ravel() # flatten the matrix into 225 array

# Get length of the training dataset
n_training = len(training)

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid):
    # Select a random number
    index = np.random.randint(0, n_training)
    # Read and display an image with selected index
    axes[i].imshow( training[index, 1:].reshape((28,28)) )
    axes[i].set_title(training[index, 0], fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()

# Prepare training and testing dataset (and normalize)
X_train = training[:, 1:] / 255
y_train = training[:, 0]

X_test = testing[:, 1:] / 255
y_test = testing[:,0]

from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=12345)

# Reshape the data to match input requirements for Neural Network
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

# import sequential, convolutional layers, max pooling, densing, flattening, dropout layers
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=32, activation='relu'))
cnn_model.add(Dense(units=10, activation='sigmoid'))

cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
epochs = 50

cnn_model.fit(X_train, y_train, batch_size=512, epochs=epochs, verbose=1, validation_data=(X_validate, y_validate))


# Evaluting Model
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))

predicted_classes = cnn_model.predict_classes(X_test)
print(predicted_classes)

# Visualize model evaluation
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
plt.show()

from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
