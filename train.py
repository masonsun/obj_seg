from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data.davis import *

# user-defined parameters
gpu_id = 0
tf.logging.set_verbosity(tf.logging.INFO)

# import data
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))

# training parameters
store_memory = True
data_aug = True
init_lr = 1e-3
boundaries = [10000, 15000, 25000, 30000, 40000]
values = [init_lr * 10**-i for i in range(len(boundaries)+1)]

# DAVIS dataset
train_file = './data/train_list.txt'
dataset = DavisDataset(train_file, None, './data/DAVIS/', 
					   store_memory=store_memory, data_aug=data_aug)

# train network
with tf.Graph().as_default():
	with tf.device('/gpu:' + str(gpu_id)):
		global_step = tf.Variable(0, name='global_step', trainable=False)
		learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
		# ... to be completed ...


if __name__ == '__main__':
	tf.app.run()