import json
from PIL import Image
import numpy as np
import tensorflow as tf

from sklearn import preprocessing


from sklearn.preprocessing import LabelBinarizer

le = preprocessing.LabelEncoder()


b = np.load("the_data.npz")
x_in = b['arr_0']
y_in = b['arr_1']
x_test = b['arr_2']
y_test = b['arr_3']
print(x_test.shape)
print(y_in.shape)
#y_in = le.fit_transform(y_in)

new_numpy = np.array([])
new_numpy1 = np.array([])
counter = 0
for i in x_in:
    print(counter)
    if(counter == 50):
        break
    normalized_arr = preprocessing.normalize(i[0:200000].reshape(-1,1))
    #normalized_arr = i[0:200000].reshape(-1,1)
    new_numpy = np.append(new_numpy, normalized_arr)
    counter += 1


counter = 0
for i in x_in:
    print(counter)
    if(counter > 100):
        break

    if(counter < 50):
        counter += 1
        continue
    print(i)
    #normalized_arr = i[0:200000].reshape(-1,1)
    normalized_arr = preprocessing.normalize(i[0:200000].reshape(-1,1))
    new_numpy1 = np.append(new_numpy1, normalized_arr)
    counter += 1

print("new_numpy1)")
print(new_numpy1.shape)
new_numpy1 = new_numpy1.reshape(51,200000)

new_numpy = new_numpy.reshape(50, 200000)
new_yin = np.array([])
new_yin1 = np.array([])
counter = 0
for iz in y_in:
    if(counter == 50):
        break
    new_yin = np.append(new_yin,iz)
    counter += 1
    print(iz)


for iz in y_in:
    if(counter > 100):
        break
    if(counter < 50):
        counter += 1
        continue

    print(i)
    new_yin1 = np.append(new_yin1, iz)
    counter += 1
    print(iz)


new_yin = new_yin.reshape(50,1)
model = tf.keras.Sequential([
 
    tf.keras.layers.Dense(units=1000, activation='softmax',input_shape=(200000,)),
    tf.keras.layers.Dense(units=1000, activation='softmax'),
    tf.keras.layers.Dense(units=1)
])
optimizer = tf.keras.optimizers.SGD()
model.compile(optimizer, loss='mse')

model.fit(new_numpy, new_yin, epochs = 200)
#model.summary()
predicted = model.predict(new_numpy)
print(predicted[0])
print(new_yin[0])
print(predicted)
print(new_yin)
#print(encoder.inverse_transform(predicted.reshape(50,75)))
#print(encoder.inverse_transform(new_yin.reshape(50,75)))

