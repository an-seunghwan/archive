#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#%%
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
#%%
red = x_train[:, :, :, 0]
green = x_train[:, :, :, 0]
blue = x_train[:, :, :, 0]

red = np.reshape(red, (1, -1))
green = np.reshape(green, (1, -1))
blue = np.reshape(blue, (1, -1))
#%%
red_dict = Counter(list(red[0]))
red_dict = sorted(dict(red_dict).items(), key=lambda x:x[0])
#%%
plt.figure(figsize=(8, 4))
plt.bar([x[0] for x in red_dict], [x[1] for x in red_dict])
plt.title('Histogram of Red')
plt.show()
#%%
green_dict = Counter(list(green[0]))
green_dict = sorted(dict(green_dict).items(), key=lambda x:x[0])
#%%
plt.figure(figsize=(8, 4))
plt.bar([x[0] for x in green_dict], [x[1] for x in green_dict])
plt.title('Histogram of Green')
plt.show()
#%%
blue_dict = Counter(list(blue[0]))
blue_dict = sorted(dict(blue_dict).items(), key=lambda x:x[0])
#%%
plt.figure(figsize=(8, 4))
plt.bar([x[0] for x in blue_dict], [x[1] for x in blue_dict])
plt.title('Histogram of Blue')
plt.show()
#%%
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
for i in range(25):
    dict_ = Counter(list(np.reshape(x_train[i][:, :, 0], (-1, ))))
    dict_ = sorted(dict(dict_).items(), key=lambda x:x[0])
    count = sum([x[1] for x in dict_])

    axes.flatten()[i].bar([x[0] for x in dict_], [x[1] / count for x in dict_])
plt.title('Histogram of Single image (Red)')
plt.show()
#%%