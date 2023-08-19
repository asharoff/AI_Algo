import json
from PIL import Image
import numpy as np
import tensorflow as tf


# Create a run config if necessary, or add the session_config to the existing
# run config.


from sklearn import preprocessing


from sklearn.preprocessing import LabelBinarizer

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


print(tf.config.list_physical_devices('GPU'))

le = preprocessing.LabelBinarizer()

from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 10
config.gpu_options.allow_growth = True


physical_devices = tf.config.list_physical_devices('GPU')


b = np.load("the_data3.npz", allow_pickle=True)

x_in = b['A']
y_in = b['B']
y_answers = b['B']
print(x_in.shape)

new_numpy = np.array([])
y_in = le.fit_transform(y_in)
print(y_in.shape)

print(x_in[0].shape)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(420, 420, 3)),
  #tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='softmax'),
  tf.keras.layers.MaxPooling2D(),
  #tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='softmax'),
  tf.keras.layers.MaxPooling2D(),
  #tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='softmax'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='softmax'),
  #tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(y_in.shape[1])
])

optimizer = tf.keras.optimizers.SGD()
model.compile(optimizer, loss='mae')

batch_size = 1
#dataset = tf.data.Dataset.from_tensor_slices((x_in, y_in))
#dataset = dataset.batch(batch_size)

#model.fit(dataset, epochs = 20000, batch_size=1)
model.fit(x_in, y_in, epochs = 20000, batch_size = 10)
predicted = model.predict(x_in)

print(predicted)
the_predicted = le.inverse_transform(predicted)
yes = 0
total = 0
for i in range(len(the_predicted)):
    if(the_predicted[i] == y_answers[i]):
        yes += 1
    total += 1

print("acurracy")
print(float(yes)/float(total))
#print(y_answers)

#print(encoder.inverse_transform(predicted.reshape(50,75)))
#print(encoder.inverse_transform(new_yin.reshape(50,75)))

