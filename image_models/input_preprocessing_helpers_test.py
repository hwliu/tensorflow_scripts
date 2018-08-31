from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.enable_eager_execution()
from absl import flags
import os
from input_preprocessing_helpers import create_input_fn_for_images_sstable
from inception_v3_on_watermark_settings import CERVALT_KEYS_TO_FEATURE_MAP
from inception_v3_on_watermark_settings import KEYS_TO_FEATURE_MAP

FLAGS = flags.FLAGS

class E2EInputParserTest(tf.test.TestCase):

  def test_validation_input_parser_with_limited_element(self):
    filename = '/media/haoweiliu/Data/tensorflow_scripts/dataset/input_test.tfrecords'

    input_fn = create_input_fn_for_images_sstable(
        filename,
        keys_to_features_map=KEYS_TO_FEATURE_MAP,
        dataset_type='TFSequenceExample',
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=4,
        do_shuffle=False,
        element_count=3)
    _, labels = input_fn()

    self.assertAllEqual(labels.shape, [3,])


if __name__ == '__main__':
  tf.test.main()
