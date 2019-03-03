import tensorflow as tf
import numpy as np

def conv3D(inputs, kernel_size, in_channels, out_channels, strides=[1, 1, 1, 1, 1]):

    w_shape=[kernel_size,kernel_size,kernel_size,in_channels,out_channels]
    W=tf.Variable(tf.truncated_normal(w_shape,stddev=0.001))
    b_shape=[out_channels]
    b=tf.Variable(tf.constant(0.01,shape=b_shape))
    z=tf.nn.conv3d(inputs,W,strides,padding='SAME')+b
    return z

def conv3d_bn_relu(inputs, kernel_size, out_channels,in_channels, strides=[1, 1, 1, 1, 1]):
    w_shape=[kernel_size,kernel_size,kernel_size,in_channels,out_channels]
    W=tf.Variable(tf.truncated_normal(w_shape,stddev=0.001))
    b_shape=[out_channels]
    b=tf.Variable(tf.constant(0.01,shape=b_shape))
    z_out=tf.nn.conv3d(inputs, W, strides, padding='SAME')
    z_out_bn=tf.contrib.layers.batch_norm(z_out, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,is_training=True)
    z_out_relu = tf.nn.relu(z_out_bn+b)
    return z_out_relu

def pad(rit_up,lef):
    lef_shape=tf.shape(lef)
    rit_up_shape=tf.shape(rit_up)
    shape1=rit_up_shape[1]-lef_shape[1]
    shape2=rit_up_shape[2]-lef_shape[2]
    shape3=rit_up_shape[3]-lef_shape[3]
    lef=tf.pad(lef,[[0,0],[shape1,0],[shape2,0],[shape3,0],[0,0]],'CONSTANT')
    return  lef
def crop(rit_up,lef):
    lef_shape=tf.shape(lef)
    rit_up_shape=tf.shape(rit_up)
    offset=[0,(rit_up_shape[1]-lef_shape[1])//2,(rit_up_shape[2]-lef_shape[2])//2,(rit_up_shape[3]-lef_shape[3])//2,0]
    size=[-1,lef_shape[1],lef_shape[2],lef_shape[3],-1]
    croped_rit_up=tf.slice(rit_up,offset,size)
    return croped_rit_up


def concat_3D(rit,lef):
	# channel fusion
    rit_concat = tf.concat([rit,lef],axis=4)
    return rit_concat


def deconv_bn_relu(inputs, kernel_size, in_channels, out_channels, activation_func=tf.nn.relu, strides=[1, 2, 2, 2, 1]):
    w_shape = [kernel_size, kernel_size, kernel_size, out_channels, in_channels]
    W = tf.Variable(tf.truncated_normal(w_shape, stddev=0.001))
    b_shape=[out_channels]
    b=tf.Variable(tf.constant(0.01,shape=b_shape))
    basic_shape = tf.stack([1, tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2, tf.shape(inputs)[3]*2, out_channels])

    deconva=tf.nn.conv3d_transpose(inputs, W, output_shape=basic_shape, strides=strides)
    deconva_bn= tf.contrib.layers.batch_norm(deconva, decay=0.9, updates_collections=None, epsilon=1e-5,scale=True, is_training=True)
    deconva_relu = tf.nn.relu(deconva_bn+b)
    return deconva_relu

def attention_module(fusion_feture,in_channels,kernel_size):
    conv_1x1 = conv3D(fusion_feture, kernel_size=1, in_channels=in_channels, out_channels=32)
    conv_kxk_1 = conv3D(conv_1x1, kernel_size=kernel_size, in_channels=32, out_channels=32)
    conv_kxk_2 = conv3D(conv_kxk_1, kernel_size=kernel_size, in_channels=32, out_channels=32)
    _softmax = tf.nn.softmax(conv_kxk_2)
    _multiply = tf.multiply(conv_1x1, _softmax)
    _fusion = concat_3D(conv_1x1, _multiply)
    _fusion_shape = _fusion.shape.as_list()
    conv_kxk_3 = conv3D(_fusion, kernel_size=kernel_size, in_channels = _fusion_shape[4], out_channels=64)
    return conv_kxk_3
