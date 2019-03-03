import numpy as np
import tensorflow as tf
import nibabel as nib
#  import self package path
from sys import path
self_package = path[0].split("/")[0:-1]
path.append(('/').join(self_package))

from src.layer import *
from input import *
import os
from src.Network import Net
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
with open('info.json') as f:
    ParSet = json.load(f)

#  cross_entropy function
def cross_entropy(label,logit):
    '''
    Loss = -target*log(softmax(logit))
    :param label: ground_truth 
    :param logit: probability score
    :return: softmax
    '''
    label = tf.cast(label, tf.int64)
    logits = tf.reshape(logit, (-1, 2))
    labels = tf.squeeze(tf.reshape(label, (-1, 1)))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
    return cross_entropy_mean

def train(sess):

    x = tf.placeholder(tf.float32, [ParSet['BATCH_SIZE'], None, None, None, ParSet['CHANNEL_NUM']], name='x_input')

    conv_aux2, conv_aux1, y_conv = Net(x)


    y_ = tf.placeholder(tf.float32,shape=(ParSet['BATCH_SIZE'],None,None,None,ParSet['CHANNEL_NUM']),name='y_input')
    y_aux2 = tf.placeholder(tf.float32, shape=(ParSet['BATCH_SIZE'], None, None, None, ParSet['CHANNEL_NUM']), name='y_aux2')
    y_aux1 = tf.placeholder(tf.float32, shape=(ParSet['BATCH_SIZE'], None, None, None, ParSet['CHANNEL_NUM']), name='y_aux1')

    with tf.name_scope('dice'):
        loss_aux2 = cross_entropy(y_aux2,conv_aux2)
        loss_aux1 = cross_entropy(y_aux1,conv_aux1)
        loss_main = cross_entropy(y_,y_conv)

        loss_total = loss_main + loss_aux1 + loss_aux2
        tf.summary.scalar('loss',loss_total)

    with tf.name_scope('train'):
        train_step=tf.train.AdamOptimizer(ParSet['LEARNING_RATE'],).minimize(loss_total)
    merged=tf.summary.merge_all()

    def feed_dict():

        data_rang = ParSet['NUM_TRAIN']
        image, label, label_aux1, label_aux2 = data_load_train(nii_index=np.random.choice(data_rang))

        return {x:image , y_:label, y_aux1:label_aux1, y_aux2:label_aux2}


    saver = tf.train.Saver()

    start_i=0
    end_i=ParSet['NUM_TRAIN']*ParSet['EPOCH']

    if eval(ParSet['IS_TRAINING_FROM_BEGIN']):
        tf.global_variables_initializer().run()
    else:
        ckpt_path = tf.train.latest_checkpoint(ParSet['CHECKPOINTS_DIR'])
        saver.restore(sess,ckpt_path)
        start_i = int(ckpt_path.split('-')[-1])
        print('Resume training from %s ,continue training...' %start_i)
    train_writer = tf.summary.FileWriter(ParSet['LOG_DIR'],sess.graph)

    for i in range(start_i,end_i):
        if i % 200 == 0:
            saver.save(sess,ParSet['CHECKPOINTS_DIR'] + 'net' ,global_step=i)
            print("have train %d step and save the checkpoits" %i)
        else:
            if (i+1) % 100 ==0:

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary,_  =sess.run([merged,train_step],feed_dict(),options=run_options,run_metadata=run_metadata)

                train_writer.add_run_metadata(run_metadata , 'step%03d' %(i+1))
                train_writer.add_summary(summary,(i+1))
            else:
                summary , _ = sess.run([merged,train_step],feed_dict=feed_dict())

    train_writer.close()

def test(sess):

    x = tf.placeholder(tf.float32, [ParSet['BATCH_SIZE'], None, None, None,ParSet['CHANNEL_NUM']], name='x_input')
    num_index = tf.placeholder(tf.uint8, shape=[])

    def feed_dict(nii_index=0):
        xs, num = load_inference(nii_index=nii_index)
        return {x: xs, num_index: num}

    _, _, _, y_conv = Net(x)

    y_softmax = tf.nn.softmax(y_conv)

    saver = tf.train.Saver()
    # checkpoints restoring
    ckpt_path = tf.train.latest_checkpoint(ParSet['CHECKPOINTS_DIR'])
    star_i = int(ckpt_path.split('-')[-1])
    print(star_i)
    print('Restoring %d checkpoints' % star_i)
    saver.restore(sess, ckpt_path)

    for i in range(ParSet['NUM_TEST']):
        pred_volume, data_nii = sess.run(y_softmax, feed_dict=feed_dict(nii_index=i))
        pred_volume = np.squeeze(pred_volume, axis=0)

        print('is processing %3dth volume...' % (i + 1))

        savename = str(data_nii + 1) + '_prob'
        y_nii = nib.Nifti1Image(pred_volume.astype(np.float), np.eye(4))
        nib.save(y_nii, ParSet['PROB_PATH'] + savename + '.nii.gz')
    print('Complete!')
