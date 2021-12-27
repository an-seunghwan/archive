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
#%%
x = tf.Variable(2., dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.abs(x - 1.)
    # y = tf.math.pow(x, 2)
#%%
tape.gradient(y, x)
#%%