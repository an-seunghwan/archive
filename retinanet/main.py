#%%
"""
Title: Object Detection with RetinaNet
Author: [Srihari Humbarwadi](https://twitter.com/srihari_rh)
Date created: 2020/05/17
Last modified: 2020/07/14
Description: Implementing RetinaNet: Focal Loss for Dense Object Detection.
Modified version
"""

"""
## Introduction

Object detection a very important problem in computer
vision. Here the model is tasked with localizing the objects present in an
image, and at the same time, classifying them into different categories.
Object detection models can be broadly classified into "single-stage" and
"two-stage" detectors. Two-stage detectors are often more accurate but at the
cost of being slower. Here in this example, we will implement RetinaNet,
a popular single-stage detector, which is accurate and runs fast.
RetinaNet uses a feature pyramid network to efficiently detect objects at
multiple scales and introduces a new loss, the Focal loss function, to alleviate
the problem of the extreme foreground-background class imbalance.

**References:**

- [RetinaNet Paper](https://arxiv.org/abs/1708.02002)
- [Feature Pyramid Network Paper](https://arxiv.org/abs/1612.03144)
"""
#%%
import os
# os.chdir('/Users/anseunghwan/Documents/GitHub/archive/retinanet')
os.chdir(r'D:\archive\retinanet')
# os.chdir('/home1/prof/jeon/an/retinanet')

# import re
# import zipfile

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from utils import *
from preprocess import *
from anchors import *
from model import *
from labeling import *
#%%
batch_size = 2
num_classes = 20
# model_dir = "model/"
epochs = 320
#%%
class_dict = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}
class_dict = {x:y-1 for x,y in class_dict.items()}
classnum_dict = {y:x for x,y in class_dict.items()}
#%%
# '''dataset'''
# train_dataset, val_dataset, test_dataset, ds_info = fetch_dataset(batch_size)
# train_dataset = train_dataset.concatenate(val_dataset)

# log_path = f'logs/voc2007'

# """
# visualization
# """
# image, bbox, label = next(iter(train_dataset))
# image = np.array(image[0], dtype=np.uint8)
# bbox = bbox[0].numpy()
# label = label[0].numpy()
# img_height, img_width, _ = image.shape

# plt.figure(figsize=(7, 7))
# # plt.axis("off")
# plt.imshow(image)
# ax = plt.gca()
# for box, _cls in zip(bbox, label):
#     text = "{}".format(classnum_dict.get(_cls))
#     x, y, w, h = box.tolist()
#     # y_min, x_min = round(y_min * img_height), round(x_min * img_width)
#     # y_max, x_max = round(y_max * img_height), round(x_max * img_width)
#     patch = plt.Rectangle(
#         [x - w/2, y - h/2], w, h, fill=False, edgecolor='red', linewidth=2
#     )
#     ax.add_patch(patch)
#     ax.text(
#         x - w/2,
#         y - h/2,
#         text,
#         bbox={"facecolor": 'blue', "alpha": 0.4},
#         clip_box=ax.clipbox,
#         clip_on=True,
#     )
# plt.show()
#%%
"""
## Implementing a custom layer to decode predictions
"""

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=20,
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, box_pred, cls_pred):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_predictions = tf.nn.sigmoid(cls_pred)
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_pred)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
#%%
"""
## Generating detections
"""
def prepare_image(image):
    ratio = 512 / tf.reduce_max(tf.shape(image)[:2])
    image_shape = [512, 512]
    image_shape = tf.cast(image_shape, dtype=tf.float32)
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio
#%%
'''dataset'''
train_dataset, val_dataset, test_dataset, ds_info = fetch_dataset(batch_size)
train_dataset = train_dataset.concatenate(val_dataset)

log_path = f'logs/voc2007'

"""
## Initializing and compiling model
"""
resnet50_backbone = get_backbone()
model = RetinaNet(num_classes, resnet50_backbone)
model.build(input_shape=[None, 512, 512, 3])
# loss_fn = RetinaNetLoss(num_classes)

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]

optimizer = K.optimizers.SGD(learning_rate=learning_rates[0], momentum=0.9)
model.summary()

label_encoder = LabelEncoder()

train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
# val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
# test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

# total_length = sum(1 for _ in train_dataset)
iteration = (ds_info.splits["train"].num_examples + ds_info.splits["validation"].num_examples) // batch_size
#%%
for epoch in range(epochs):
    loss_avg = tf.keras.metrics.Mean()
    loss_box_avg = tf.keras.metrics.Mean()
    loss_clf_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    train_iter = iter(train_dataset)
    test_iter = iter(test_dataset)
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        '''learning rate schedule'''
        epoch_ = epoch * iteration + batch_num
        if epoch_ < learning_rate_boundaries[0]: optimizer.lr = learning_rates[0]
        elif epoch_ < learning_rate_boundaries[1]: optimizer.lr = learning_rates[1]
        elif epoch_ < learning_rate_boundaries[2]: optimizer.lr = learning_rates[2]
        elif epoch_ < learning_rate_boundaries[3]: optimizer.lr = learning_rates[3]
        elif epoch_ < learning_rate_boundaries[4]: optimizer.lr = learning_rates[4]
        else: optimizer.lr = learning_rates[5]

        image, bbox, label = next(train_iter)
        image, bbox_true, cls_true = label_encoder.encode_batch(image, bbox, label)        

        with tf.GradientTape(persistent=True) as tape:
            bbox_pred, cls_pred = model(image)
            loss, box_loss, clf_loss = RetinaNetLoss(bbox_true, cls_true, bbox_pred, cls_pred)
                        
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        
        loss_avg(loss)
        loss_box_avg(box_loss)
        loss_clf_avg(clf_loss)
        accuracy(cls_true, cls_pred)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'Box': f'{loss_box_avg.result():.4f}',
            'CLS': f'{loss_clf_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    
    with train_writer.as_default():
        tf.summary.scalar('loss', loss_avg.result(), step=epoch)
        tf.summary.scalar('loss_box', loss_box_avg.result(), step=epoch)
        tf.summary.scalar('loss_clf', loss_clf_avg.result(), step=epoch)
        tf.summary.scalar('accuracy', accuracy.result(), step=epoch)

    # Reset metrics every epoch
    loss_avg.reset_states()
    loss_box_avg.reset_states()
    loss_clf_avg.reset_states()
    accuracy.reset_states()
    
    if epoch % 50 == 0:
        """
        save model
        """
        model_path = f'{log_path}/{current_time}'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_weights(model_path + '/model_{}_epoch{}.h5'.format(current_time, epoch), save_format="h5")
        
    if epoch % 3 == 0:
        """
        ## Building inference model
        """
        image = K.Input(shape=[512, 512, 3], name="image")
        predictions = model(image, training=False)
        detections = DecodePredictions(confidence_threshold=0.5)(image, predictions[0], predictions[1])
        inference_model = K.Model(inputs=image, outputs=detections)
        
        """
        save results
        """
        flag = 0
        results = []
        for sample in test_iter:
            flag += 1
            image = tf.cast(sample[0], dtype=tf.float32)
            input_image, ratio = prepare_image(image[0])
            detections = inference_model.predict(input_image)
            num_detections = detections.valid_detections[0]
            class_names = [
                classnum_dict.get(int(x)) for x in detections.nmsed_classes[0][:num_detections]
            ]
            fig = visualize_detections(
                image[0],
                detections.nmsed_boxes[0][:num_detections] / ratio,
                class_names,
                detections.nmsed_scores[0][:num_detections],
            )
            fig.canvas.draw()
            results.append(np.array(fig.canvas.renderer._renderer))
            if flag == 10: break
        
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for i in range(len(results)):
            axes.flatten()[i].imshow(results[i])
            axes.flatten()[i].axis('off')
        plt.tight_layout()
        plt.savefig('{}/sample_{}.png'.format(model_path, epoch),
                    dpi=200, bbox_inches="tight", pad_inches=0.1)
        # plt.show()
        plt.close()
#%%
"""
save model
"""
model_path = f'{log_path}/{current_time}'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save_weights(model_path + '/model_{}.h5'.format(current_time), save_format="h5")
#%%
# model_path = log_path + '/20220210-205432'
# model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
# resnet50_backbone = get_backbone()
# model = RetinaNet(num_classes, resnet50_backbone)
# model.build(input_shape=[None, 512, 512, 3])
# model.load_weights(model_path + '/' + model_name)
# model.summary()
#%%
"""
## Building inference model
"""
image = tf.keras.Input(shape=[512, 512, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.1)(image, predictions[0], predictions[1])
inference_model = tf.keras.Model(inputs=image, outputs=detections)
#%%
"""
save results
"""
flag = 0
results = []
for sample in test_iter:
    flag += 1
    image = tf.cast(sample[0], dtype=tf.float32)
    input_image, ratio = prepare_image(image[0])
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        classnum_dict.get(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    fig = visualize_detections(
        image[0],
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
    fig.canvas.draw()
    results.append(np.array(fig.canvas.renderer._renderer))
    if flag == 10: break

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
for i in range(len(results)):
    axes.flatten()[i].imshow(results[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/sample_{}.png'.format(model_path, epoch),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%