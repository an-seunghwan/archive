#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
#%%
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)
#%%
input_layer = layers.Input(x_train.shape[1:])
conv1 = layers.Conv2D(16, 4, 2, padding='same', activation='relu')
conv2 = layers.Conv2D(32, 4, 2, padding='same', activation='relu')
conv3 = layers.Conv2D(64, 4, 2, padding='same', activation='relu')
output_layer = layers.Dense(10, activation='softmax')

h1 = conv1(input_layer)
h2 = conv2(h1)
h3 = conv3(h2)
output = output_layer(layers.GlobalAveragePooling2D()(h3))

model = tf.keras.Model(inputs=input_layer, outputs=output)
model.summary()
#%%
reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(h1), axis=[1, 2, 3]))
reg_loss += tf.reduce_mean(tf.reduce_sum(tf.square(h2), axis=[1, 2, 3]))
reg_loss += tf.reduce_mean(tf.reduce_sum(tf.square(h3), axis=[1, 2, 3]))
lambda_ = 0.1
model.add_loss(lambda_ * reg_loss)
#%%
model.compile(optimizer='adam',
            loss=K.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])
#%%
model_history = model.fit(x=x_train, y=y_train_onehot, 
                        epochs=20,
                        validation_split=0.2)
#%%