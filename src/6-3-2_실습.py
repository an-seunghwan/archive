#%%
import tensorflow as tf
from tensorflow.keras import layers
#%%
# layer는 재귀적으로 전진 방향 전파 학습을 하는 도중 손실함수 값을 수집한다!
# layer에서 `call` method는 손실 값을 저장하는 tensor를 생성할 수 있도록 해주어, 후에 training loop을 작성할 때 사용가능하도록 해준다.
# `self.add_loss(value)`를 사용!
#%%
class ActivityRegularizationLayer(layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate
    
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
#%%
# 이렇게 생성된 손실 값은(임의의 내부 layer의 손실 함수 값을 포함하여) `layer.losses`를 이용해 불러올 수 있다. 
# 이러한 특성은 top-level layer에서의 모든 `__call__`의 시작에 초기화 된다. 
# 이는 `layers.losses`가 항상 마지막 전진 방향 전파 학습의 손실 값만을 저장하기 위함이다.

class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activitiy_reg = ActivityRegularizationLayer(1e-2)
    
    def call(self, inputs):
        return self.activitiy_reg(inputs)
#%%
layer = OuterLayer()
'''어떠한 layer도 call되지 않았으므로 손실 값이 없다'''
assert len(layer.losses) == 0
_ = layer(tf.random.normal((1, 1)))
'''layer가 1번 call되었으므로 손실 값은 1개'''
print(layer.losses)
assert len(layer.losses) == 1
#%%
'''layer.losses는 각각의 __call__의 시작에서 초기화'''
_ = layer(tf.random.normal((1, 1)))
'''마지막으로 생성된 손실 값'''
print(layer.losses)
assert len(layer.losses) == 1
#%%