import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import random
from progressbar import progressbar
from sklearn.cross_validation import train_test_split

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline

def build_fc_model():
    fc_model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, input_dim=784, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    return fc_model

def build_cnn_model():
    cnn_model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(filters=24, kernel_size=(3,3), input_shape=(28, 28, 1), activation=tf.nn.relu),       
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=36, kernel_size=(3,3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, input_dim=784, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return cnn_model

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = np.expand_dims(train_images, axis=-1)/255.
train_labels = np.int64(train_labels)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_images = np.expand_dims(test_images, axis=-1)/255.
test_labels = np.int64(test_labels)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.33, random_state=42)

plt.figure(figsize=(10,10))
random_inds = np.random.choice(60000,36)
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])
    
fc_model = build_fc_model()
BATCH_SIZE = 64
EPOCHS = 5

model.compile(optimizer=tf.train.GradientDescentOptimizer(.1), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, verbose = 0, epochs=EPOCHS, batch_size=BATCH_SIZE)

score = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)

test_acc = score[1]

print('Test accuracy:', test_acc)

cnn_model = build_cnn_model()
print(cnn_model.summary())

cnn_model.compile(optimizer=tf.train.AdadeltaOptimizer(10), loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit(train_images, train_labels, verbose = 0, epochs=EPOCHS, batch_size=BATCH_SIZE)

score = cnn_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)

test_acc = score[1]

print('Test accuracy:', test_acc)

predictions = cnn_model.predict(test_images)

predictions[0]

predictions = fc_model.predict(test_images)

predictions[0]

test_labels[0]

plt.imshow(np.squeeze(test_images[0]), cmap=plt.cm.binary)