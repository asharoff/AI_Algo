import json
from PIL import Image
import numpy as np
import tensorflow as tf


from sklearn import preprocessing


from sklearn.preprocessing import LabelBinarizer

le = preprocessing.LabelBinarizer()


b = np.load("the_data3.npz")
print(b)
x_in = b['arr_0']
y_in = b['arr_1']
y_z = y_in
x_test = b['arr_2']
y_test = b['arr_3']

y_inn = np.array([])
for i in y_in:
    for q in i:
        if(q == 1):
            print("here")

counter = 0
the_yinn = np.array([])
y_in = le.fit_transform(y_z)
for i in y_in:
    if(counter == 50):
        break
    the_yinn = np.append(the_yinn, i)
    counter += 1


print(the_yinn.shape)
the_yinn = the_yinn.reshape(50,59)
print(the_yinn)
new_numpy = np.array([])
new_numpy1 = np.array([])
counter = 0
for i in x_in:
    if(counter == 50):
        break
    normalized_arr = preprocessing.normalize(i[0:200000].reshape(-1,1))
    #normalized_arr = i[0:200000].reshape(-1,1)
    new_numpy = np.append(new_numpy, normalized_arr)
    counter += 1


counter = 0
for i in x_in:
    if(counter > 100):
        break

    if(counter < 50):
        counter += 1
        continue
    #normalized_arr = i[0:200000].reshape(-1,1)
    normalized_arr = preprocessing.normalize(i[0:200000].reshape(-1,1))
    new_numpy1 = np.append(new_numpy1, normalized_arr)
    counter += 1

new_numpy1 = new_numpy1.reshape(51,200000)

new_numpy = new_numpy.reshape(50, 200000)
new_yin = np.array([])
new_yin1 = np.array([])
counter = 0


model = tf.keras.Sequential([
 
    tf.keras.layers.Dense(units=1000, activation='softmax',input_shape=(200000,)),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=59)
])
optimizer = tf.keras.optimizers.SGD()
model.compile(optimizer, loss='mse')

model.fit(new_numpy, the_yinn, epochs = 200)
#model.summary()
predicted = model.predict(new_numpy)

print(predicted)
print(le.inverse_transform(predicted.reshape(50,59)))

#print(encoder.inverse_transform(predicted.reshape(50,75)))
#print(encoder.inverse_transform(new_yin.reshape(50,75)))

