"""Settings for inception V3."""
import numpy
import tensorflow as tf

INPUT_FEATURE_NAME = 'image_bytes'

DATASET_IMAGE_KEY = 'image_feature_0'

DATASET_LABEL_KEY = 'image_0_image_target_probs'

KEYS_TO_FEATURE_MAP = {
    DATASET_IMAGE_KEY:
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    DATASET_LABEL_KEY:
        tf.FixedLenFeature(
            [10], dtype=tf.int64, default_value=[0 for _ in xrange(10)]),
}

#  To parse Cervalat dataset
CERVALT_DATASET_IMAGE_KEY = 'image/encoded'
CERVALT_DATASET_LABEL_KEY = 'image/class/label'
CERVALT_KEYS_TO_FEATURE_MAP = {
    CERVALT_DATASET_IMAGE_KEY:
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    CERVALT_DATASET_LABEL_KEY:
        tf.FixedLenFeature(
            [1], dtype=tf.int64, default_value=[0]),
}

INCEPTION_V3_TARGET_IMAGE_SIZE = numpy.array([299, 299])
INCEPTION_V3_MODULE_PATH = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
