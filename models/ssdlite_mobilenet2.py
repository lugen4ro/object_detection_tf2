#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSD Object Detector & MobilenetV2 Classifier
"""
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax, DepthwiseConv2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.nn import relu, relu6, softmax

import numpy as np
import tensorflow as tf

def Prediction(feature_map, num_priors, num_classes, img_size, min_size, max_size, aspect):
    
    ### Localization
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(feature_map)
    x = BatchNormalization()(x)
    x = relu6(x)
    
    # Pointwise
    # num_priors*4 filters because each box is described by 4 values
    x = Conv2D(filters=num_priors*4, kernel_size=1, padding='same', use_bias=False)(x)
    flat_boxes = Flatten()(x)
    
    
    ### Classification
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(feature_map)
    x = BatchNormalization()(x)
    x = relu6(x)
    
    # Pointwise
    x = Conv2D(filters=num_priors*num_classes, kernel_size=1, padding='same', use_bias=False)(x)
    flat_classes = Flatten()(x)
    
    ### Priorboxes 
    prior_boxes = PriorBox(x, img_size, min_size=min_size, max_size=max_size, aspect=aspect, variances=[0.1,0.1,0.2,0.2])
    
    return flat_boxes, flat_classes, prior_boxes


def LiteConv(x, i, filters):
    ''' Mobilenet Lite Convolution '''
    
    # Expand
    x = Conv2D(filters=filters//2, kernel_size=1 , padding='same', use_bias=False, name='LiteConv_' + str(i) + 'expand')(x)
    x = BatchNormalization(name='LiteConv_' + str(i) + 'expand' + '_BN')(x)
    x = relu6(x, name='LiteConv_' + str(i) + 'expand' + '_ReLU')
    
    # Depthwise (Chagned padding from same to valid because it is only half the spatial size)
    x = DepthwiseConv2D(kernel_size=3, strides=2, use_bias=False, padding='same', name='LiteConv_' + str(i) + 'depthwise')(x)
    x = BatchNormalization(name='LiteConv_' + str(i) + 'depthwise' + '_BN')(x)
    x = relu6(x, name='LiteConv_' + str(i) + 'depthwise' + '_ReLU')
    
    # Project
    x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name='LiteConv_' + str(i) + 'project')(x)
    x = BatchNormalization(name='LiteConv_' + str(i) + 'project' + '_BN')(x)
    x = relu6(x, name='LiteConv_' + str(i) + 'project' + '_ReLU')

    return x


def SSDLite_Mobilenet2(size=None, channels=3, num_anchors=5, num_classes=20, training=False):
    x = inputs = Input([size, size, channels], name='input')
    
    # MobilenetV2 Backbone
    # 155 layers (without top means no final global average pool and dense layer)
    Mobilenet2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    #x = Mobilenet2(x)

    # outputs used from Backbone.
    # The whole model has 16 blocks with an additional normal Conv layer at the end
    # We use the output from after the 13th block and the very end
    out1 = Mobilenet2.get_layer('block_13_project_BN').output # (None, 7, 7, 160)
    out2 = Mobilenet2.output # (None, 7, 7, 1280)
    
    
    # Additional 4 LiteConv Layers (Inverted Residual Blocks)
    # Maybe get tensor before batchnorm and relu of last layer to pass to prediction layer
    out3 = LiteConv(out2, 1, 512)
    out4 = LiteConv(out3, 2, 256)
    out5 = LiteConv(out4, 3, 128) # Other implementation makes this 256
    out6 = LiteConv(out5, 4, 128)
    
    
    # Apply SSD prediction layers
    flat_boxes1, flat_classes1, prior_boxes1 = Prediction(out1, 3, num_classes, img_size=(size,size), min_size=60.0, max_size=None , aspect=[2])
    flat_boxes2, flat_classes2, prior_boxes2 = Prediction(out2, 6, num_classes, img_size=(size,size), min_size=105.0, max_size=150.0 , aspect=[2, 3])
    flat_boxes3, flat_classes3, prior_boxes3 = Prediction(out3, 6, num_classes, img_size=(size,size), min_size=150.0, max_size=195.0 , aspect=[2, 3])
    flat_boxes4, flat_classes4, prior_boxes4 = Prediction(out4, 6, num_classes, img_size=(size,size), min_size=195.0, max_size=240.0 , aspect=[2, 3])
    flat_boxes5, flat_classes5, prior_boxes5 = Prediction(out5, 6, num_classes, img_size=(size,size), min_size=240.0, max_size=285.0 , aspect=[2, 3])
    flat_boxes6, flat_classes6, prior_boxes6 = Prediction(out6, 6, num_classes, img_size=(size,size), min_size=285.0, max_size=300.0 , aspect=[2, 3])

        
    # Gather all predictions
    boxes = tf.concat([flat_boxes1, flat_boxes2, flat_boxes3, flat_boxes4, flat_boxes5, flat_boxes6], axis=1)
    classes = tf.concat([flat_classes1, flat_classes2, flat_classes3, flat_classes4, flat_classes5, flat_classes6], axis=1)
    priors = tf.concat([prior_boxes1, prior_boxes2, prior_boxes3, prior_boxes4, prior_boxes5, prior_boxes6], axis=1)
    
    # get total number of boxes
    num_boxes = boxes.shape[-1] // 4
     
        
    # reshape and get confidence scores with softmax
    boxes = tf.reshape(boxes, (num_boxes, 4))
    classes = tf.reshape(classes, (num_boxes, num_classes))
    classes = softmax(classes, axis=-1)
    priors = tf.reshape(priors, (num_boxes, 8))
    
    # output predictions
    predictions = tf.concat([boxes, classes, priors], axis=1)
    
    return Model(inputs, predictions, name='SSDLite_MobilenetV2')

    
def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model

    """
    model(tf.random.uniform((1, 300, 300, 3)))