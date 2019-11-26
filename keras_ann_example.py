import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

mnist = tf.keras.datasets.mnist
# 28x28 grayscale image of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

 # feature normalization (all features between 0 - 1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
 
# build the feed forward model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # 28x28 -> 784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # hidden
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # hidden
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3)

validation_loss, validation_accuracy = model.evaluate(x_test, y_test)
print(validation_accuracy)

# make a prediction
prediction = model.predict([x_train[:1]])
print(np.argmax(prediction[0]))
plt.imshow(x_train[:1])
