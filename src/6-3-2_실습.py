---
title: "Custom modeling with Keras (2)"
excerpt: loss, config and serialization
toc: true
toc_sticky: true

author_profile: false

date: 2019-12-30 16:30:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---

> 이 글은 다음 문서를 참조하고 있습니다!
> [https://www.tensorflow.org/guide/keras/custom_layers_and_models](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session() # 간단한 초기화 방법(노트북 환경에서)
```
* What is ```__future__```? (coming soon!)

## **layer는 재귀적으로 전진 방향 전파 학습을 하는 도중 손실함수 값을 수집한다!**

layer에서 `call` method는 손실 값을 저장하는 tensor를 생성할 수 있도록 해주어, 후에 training loop을 작성할 때 사용가능하도록 해준다.
→ `self.add_loss(value)`를 사용!

```python
class ActivityRegularizationLayer(layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate
    
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
```

이렇게 생성된 손실 값은(임의의 내부 layer의 손실 함수 값을 포함하여) `layer.losses`를 이용해 불러올 수 있다. 이러한 특성은 top-level layer에서의 모든 `__call__`의 시작에 초기화 된다. 이는 `layers.losses`가 항상 마지막 전진 방향 전파 학습의 손실 값만을 저장하기 위함이다.
```python
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activitiy_reg = ActivityRegularizationLayer(1e-2)
    
    def call(self, inputs):
        return self.activitiy_reg(inputs)
```
```python
layer = OuterLayer()
'''어떠한 layer도 call되지 않았으므로 손실 값이 없다'''
assert len(layer.losses) == 0
_ = layer(tf.zeros(1, 1))
'''layer가 1번 call되었으므로 손실 값은 1개'''
assert len(layer.losses) == 1
'''layer.losses는 각각의 __call__의 시작에서 초기화'''
_ = layer(tf.zeros(1, 1))
'''마지막으로 생성된 손실 값'''
assert len(layer.losses) == 1
```
추가로, `loss` 특성은 임의의 내부 layer에서 생성된 정규화 손실 값 또한 포함한다.
```python
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    
    def call(self, inputs):
        return self.dense(inputs)
```
```python
layer = OuterLayer()
_ = layer(tf.zeros((1, 1)))
print(layer.dense.kernel)
print(tf.reduce_sum(layer.dense.kernel) ** 2)
'''
이 값은 1e-3 * sum(layer.dense.kernel ** 2)와 같다
(이 손실은 kernel_regularizer에 의해 생성)
'''
print(layer.losses)
```
```
<tf.Variable 'outer_layer_3/dense_1/kernel:0' shape=(1, 32) dtype=float32, numpy=
array([[-0.17390427, -0.33782747,  0.00282753, -0.3532169 ,  0.34208316,
        -0.37428847, -0.05844164,  0.01640856,  0.32005012,  0.3649932 ,
         0.35369265,  0.20181292,  0.23604548, -0.2578826 ,  0.09839004,
        -0.18697263,  0.0741716 , -0.06126347,  0.4143306 , -0.16958284,
         0.08949876,  0.2845322 , -0.26741046,  0.32063776,  0.15464622,
         0.37672937,  0.3461277 ,  0.00118405, -0.15776005, -0.14735147,
        -0.3484411 , -0.25038716]], dtype=float32)>
tf.Tensor(0.7283453, shape=(), dtype=float32)
[<tf.Tensor: id=119, shape=(), dtype=float32, numpy=0.0020875973>]
```

training loop에 응용: [https://www.tensorflow.org/guide/keras/train_and_evaluate](https://www.tensorflow.org/guide/keras/train_and_evaluate) (coming soon!)
