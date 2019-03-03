import numpy as np
import tensorflow as tf
from .layer import *
import json
import pdb
with open('info.json') as f:
    ParSet = json.load(f)
pdb.set_trace()

def Net(x):
	# f1
	conv1_1 = conv3D(x, kernel_size=3, in_channels=1, out_channels=32)
	conv1_1_bn = tf.contrib.layers.batch_norm(conv1_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv1_1_relu = tf.nn.relu(conv1_1_bn)

	conv1_2 = conv3D(conv1_1_relu, kernel_size=3, in_channels=32, out_channels=32)
	conv1_2_bn = tf.contrib.layers.batch_norm(conv1_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)

	conv1_2_relu = tf.nn.relu(conv1_2_bn)

	conv1_2_relu_attention = attention_module(conv1_2_relu, in_channels=32, kernel_size=3)

	pool1 = tf.layers.max_pooling3d(inputs=conv1_2_relu, pool_size=2, strides=2, padding='same')
	# f2
	conv2_1 = conv3D(pool1, kernel_size=3, in_channels=32, out_channels=64)
	conv2_1_bn = tf.contrib.layers.batch_norm(conv2_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv2_1_relu = tf.nn.relu(conv2_1_bn)

	conv2_2 = conv3D(conv2_1_relu, kernel_size=3, in_channels=64, out_channels=64)
	conv2_2_bn = tf.contrib.layers.batch_norm(conv2_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv2_2_relu = tf.nn.relu(conv2_2_bn)

	conv2_2_relu_attention = attention_module(conv2_2_relu, in_channels=64, kernel_size=3)

	pool2 = tf.layers.max_pooling3d(inputs=conv2_2_relu, pool_size=2, strides=2, padding='same')

	# f3
	conv3_1 = conv3D(pool2, kernel_size=5, in_channels=64, out_channels=64)
	conv3_1_bn = tf.contrib.layers.batch_norm(conv3_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv3_1_relu = tf.nn.relu(conv3_1_bn)

	conv3_2 = conv3D(conv3_1_relu, kernel_size=5, in_channels=64, out_channels=64)
	conv3_2_bn = tf.contrib.layers.batch_norm(conv3_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv3_2_relu = tf.nn.relu(conv3_2_bn)
	conv3_2_relu_attention = attention_module(conv3_2_relu, in_channels=64, kernel_size=5)

	pool3 = tf.layers.max_pooling3d(inputs=conv3_2_relu, pool_size=2, strides=2, padding='same')

	# f4
	conv4_1 = conv3D(pool3, kernel_size=5, in_channels=64, out_channels=64)
	conv4_1_bn = tf.contrib.layers.batch_norm(conv4_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv4_1_relu = tf.nn.relu(conv4_1_bn)

	conv4_2 = conv3D(conv4_1_relu, kernel_size=5, in_channels=64, out_channels=64)
	conv4_2_bn = tf.contrib.layers.batch_norm(conv4_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv4_2_relu = tf.nn.relu(conv4_2_bn)


	# conv_aux3 = conv3D(conv4_2_relu, kernel_size=1, in_channels=128, out_channels=2)
	# up1
	deconv3_1 = deconv_bn_relu(conv4_2_relu, kernel_size=5, in_channels=64, out_channels=64)
	concat_3 = concat_3D(deconv3_1, conv3_2_relu_attention)
	attention_3 = attention_module(concat_3, in_channels=128, kernel_size=5)
	# auxiliary prediction 3
	conv_aux2 = conv3D(attention_3, kernel_size=1, in_channels=64, out_channels=2)

	# up2
	deconv2_1 = deconv_bn_relu(attention_3, kernel_size=5, in_channels=64, out_channels=64)
	concat_2 = concat_3D(deconv2_1, conv2_2_relu_attention)
	attention_2 = attention_module(concat_2, in_channels=128, kernel_size=5)
	# auxiliary prediction 2
	conv_aux1 = conv3D(attention_2, kernel_size=1, in_channels=64, out_channels=2)

	# up1
	deconv1_1 = deconv_bn_relu(attention_2, kernel_size=3, in_channels=64, out_channels=32)
	concat_1 = concat_3D(deconv1_1, conv1_2_relu_attention)
	attention_1 = attention_module(concat_1, in_channels=96, kernel_size=3)
	y_conv = conv3D(attention_1, kernel_size=1, in_channels=64, out_channels=2)

	return conv_aux2, conv_aux2, y_conv




