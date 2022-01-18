#%%
import tensorflow as tf
from tensorflow.keras import layers
#%%
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                                                  trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  # 위와 같이 shape을 적용하면 column에 더하기 아님
                                                  # 각 units에 값이 더해진다고 생각!
                                                  dtype='float32'),
                                                  trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
#%%
x = tf.random.normal(shape=(3,3))
linear_layer = Linear(5, 3)
y = linear_layer(x)
print(linear_layer.b)
print(y)
print(tf.matmul(x, linear_layer.w))
print(tf.matmul(x, linear_layer.w) + linear_layer.b)
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
#%%
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units, ),
                                 initializer='zeros',
                                 trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
#%%
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
print('trainable_weights:', linear_layer.trainable_weights)
#%%
class Compute_Sum(layers.Layer):
    def __init__(self, input_dim):
        super(Compute_Sum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim, )),
                                 trainable=False)
    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0)) 
        # 지금까지 입력된 값들을 total에 상수처럼 누적하여 저장
        return self.total
#%%
x = tf.ones((2, 2))
my_sum = Compute_Sum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
#%%
print('weights:', len(my_sum.weights))
print('non-trainable weights:', len(my_sum.non_trainable_weights))
print('trainable_weights:', my_sum.trainable_weights)
#%%
#%%
# 많은 경우에, input의 크기를 미리 알 수 없는 경우가 있고, layer를 만든 이후에 이러한 input 값이 알려지면 weights를 생성하고 싶을 수 있다.
# Keras API에서는, `build(inputs_shape)` method를 이용해 다음과  같이 weights를 이후에 생성할 수 있다.
# `__call__` method는 첫 번째 호출이 되는 시점에 자동으로 `build`를 실행시킨다.

class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units
    def build(self, input_shape): 
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units, ),
                                 initializer='random_normal',
                                 trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
#%%
x = tf.ones((3, 3))        
linear_layer = Linear(units=12) # 객체를 할당하는 시점에, 어떠한 input이 사용될 지 모른다.
y = linear_layer(x) # layer의 weights는 동적으로 처음으로 호출되는 시점에 생성된다.
print(y)
#%%