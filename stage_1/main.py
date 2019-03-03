import numpy as np
import tensorflow as tf

import json

info = {
    "LEARNING_RATE" : 1e-5,
    "LOG_DIR" : "./log/",
    "CHECKPOINTS_DIR" : "./checkpoints/",
    "IS_TRAINING_FROM_BEGIN" : "True",
    "BATCH_SIZE": 1,
    "CHANNEL_NUM" : 1,
    "PHASE": "train",
    "NUM_TRAIN":2112,
    "EPOCH": 30,
    "NUM_TEST" : 50,
    "SAVE_PATH": " ",
    "TRAIN_PATH": " ",
    "TEST_PATH": " ",
    "PROB_PATH": " "
}
with open('info.json','w') as f:
    json.dump(info,f,sort_keys=True,indent=4)

from USegNet-DS-RE import train,test

with open('info.json') as f:
    ParSet=json.load(f)

def main():
    # GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if ParSet['PHASE']=='train':
            train(sess)
        if ParSet['PHASE'] == 'test':
            test(sess)

if __name__ == '__main__':
    tf.app.run(main=main())




