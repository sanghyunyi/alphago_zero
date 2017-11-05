import tensorflow as tf
from tensorflow.contrib.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "0"
K.set_session(tf.Session(config=config))

from AlphagoZero.training.optimization import run_optimization

run_optimization()
