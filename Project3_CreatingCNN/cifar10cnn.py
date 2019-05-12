# TensorFlow Project
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net
#
# Project Description:
# Design a convolutional neural network in TensorFlow to train a CNN
# to recognize images from the classical cifar10 dataset.  The training
# and testing should take less than 8 minutes and the model should
# accurately predict images with an accuracy greater than 64%.


import datetime
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from numpy.random import seed
from tensorflow import set_random_seed

# Epoch and batch size variables
EPOCH = 40
BATCH_SIZE = 500

# Set random seed to 1 to make sure results are reproducible
seed(1)
set_random_seed(1)

# Load the cifar10 dataset
cifar10 = keras.datasets.cifar10
(X_train, y_train),(X_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create model
model = keras.models.Sequential([

    keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.4),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=tf.nn.softmax)

])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Start timer prior to training and testing
t_start = datetime.datetime.now()

print('Pre-process Training Data:')
X_train = X_train /255.0

print('Training:')
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH)

print('Preprocess Testing Data:')
X_test = X_test /255.0
print('Testing:')
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:{}, Test Loss:{} '.format(test_acc, test_loss))

print("Predicting the class for some sample test data:")
prob_result = model.predict(X_test[0:25])
class_result = prob_result.argmax(axis = -1)
print(class_result.shape)
plt.figure("CFAR10 sample test results",figsize=(12, 12))

time_dur = round((datetime.datetime.now()-t_start).total_seconds()/60.0, 2)

# Update log file with results to compare different models
print("Program execution time: ", time_dur, " mins")
f = open("results.txt", "a+")
f.write("\n\nRun Summary:\n")
f.write('Test accuracy:{}, Test Loss:{}\n'.format(test_acc, test_loss))
f.write('Epoch:{}, Batch size:{}\n'.format(EPOCH, BATCH_SIZE))
f.write("Total time: {} mins\n" .format(time_dur))
model.summary(print_fn=lambda x: f.write(x + '\n'))

# Display first 25 images from the cifar10 dataset
# along with class label and predicted class label
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    label = '{} as {}'.format(class_names[y_test[i,0]], class_names[class_result[i]])
    plt.xlabel(label)
plt.show()