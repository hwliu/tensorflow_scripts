from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
#from google3.learning.brain.contrib.slim.nets.inception_v3 import inception_v3
from PIL import Image
import PIL

FLAGS = flags.FLAGS

flags.DEFINE_string('output_model_dir', '/tmp/inception_on_fmnist',
                    'The directory to save the trained model.')

flags.DEFINE_float('dropout_rate', 0.2,
                   'Dropout rate to use when training the network.')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate to use when training the network.')

flags.DEFINE_integer('training_step', 200,
                     'Number of training steps.')

flags.DEFINE_integer('batch_size', 100,
                     'Batch size for training/eval.')

def main(unused_argv):
  run_config = tf.estimator.RunConfig(save_summary_steps=10)

  return 0



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
