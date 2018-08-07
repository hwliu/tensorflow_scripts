import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys


def main(unused_argvs):
  TENSORFLOW_PATH = '/media/haoweiliu/Data/tensorflow/models'
  models_path = os.path.join(TENSORFLOW_PATH, 'models')
  sys.path.append(models_path)

  from official.wide_deep import census_dataset
  from official.wide_deep import census_main
  census_dataset.download("./dataset/")

  return 0

if __name__ == '__main__':
  tf.app.run(main)
