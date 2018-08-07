"""Utility functions to process the input for inception v3 model."""
import tensorflow as tf
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import DATASET_IMAGE_KEY
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import DATASET_LABEL_KEY
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INPUT_FEATURE_NAME
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import KEYS_TO_FEATURE_MAP
from inception_v3_on_watermark_settings import DATASET_IMAGE_KEY
from inception_v3_on_watermark_settings import DATASET_LABEL_KEY
from inception_v3_on_watermark_settings import INPUT_FEATURE_NAME
from inception_v3_on_watermark_settings import KEYS_TO_FEATURE_MAP


def shardedfile_to_filelist(filename):
  """Convert sharded file name into a list of file names."""
  if '@' not in filename:
    return filename

  prefix, num_fileshards = filename.rsplit('@', 1)
  if not num_fileshards.isdigit():
    return filename

  num_fileshards = int(num_fileshards)
  namelist = [
      ''.join((prefix, '-%05d-of-%05d' % (i, num_fileshards)))
      for i in xrange(0, num_fileshards)
  ]
  return namelist


def crop_and_resize_image(image, target_image_size, central_fraction=0.875):
  """Crops and rescals the input image."""
  # Crop the central region of the image.
  image = tf.image.central_crop(image, central_fraction)
  # Resize the image to the target height and width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(image, target_image_size)
  image = tf.squeeze(image, [0])
  return image


def preprocess(image, target_image_size):
  """Preprocesses the image by type convering, resizing and rescaling."""
  # Scale image and convert to float32. image is of type uint8.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = crop_and_resize_image(image, target_image_size)
  image = tf.multiply(tf.subtract(image, 0.5), 2.0)
  return image


def preprocess_image(img_encoded, target_image_size):
  """Decodes raw string into image and processes it."""
  image = tf.image.decode_image(img_encoded, 3)
  image.set_shape([None, None, 3])
  image = preprocess(image, target_image_size)
  return image


def preprocess_label(label):
  """Returns 1 if one of the elements is set: 0, 2, 4, 7, 9 in label."""
  # see https://cs.corp.google.com/piper///depot/google3/experimental/users/bingjian/images/tools/flume_convert_from_sequence_example.cc?ws=haoweiliu/871
  label = tf.gather(tf.cast(label, tf.int32), [0, 2, 4, 7, 9])
  output_label = tf.cast(tf.reduce_any(tf.greater(label, 0)), tf.int32)
  return output_label


def filter_empty_feature(value, keys_to_features):
  context, _ = tf.parse_single_sequence_example(
      value, context_features=keys_to_features)

  raw_bytes = tf.decode_raw(context[DATASET_IMAGE_KEY], tf.uint8)
  return tf.greater(tf.size(raw_bytes), 0)


def parser(value, keys_to_features):
  """Parses the raw data from SSTable into image and labels."""
  context, _ = tf.parse_single_sequence_example(
      value, context_features=keys_to_features)

  image = {INPUT_FEATURE_NAME: context[DATASET_IMAGE_KEY]}
  label = preprocess_label(context[DATASET_LABEL_KEY])
  return image, label


def create_input_fn_for_images_sstable(input_file_name,
                                       mode,
                                       batch_size=50,
                                       do_shuffle=True,
                                       do_filter=True,
                                       num_parallelism=128,
                                       queue_capacity=256,
                                       element_count=-1,
                                       skip_count=0):
  """Creates and returns an input_fn based on input parameters."""
  keys_to_features = KEYS_TO_FEATURE_MAP
  def input_fn():
    """Creates an input_fn based on input parameters."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if is_training:
      num_epochs = None
    else:
      num_epochs = 1

    parse_fn = (lambda x: parser(x, keys_to_features))
    filter_fn = (lambda x: filter_empty_feature(x, keys_to_features))

    input_shard_file_list = shardedfile_to_filelist(input_file_name)
    with tf.name_scope('read_batch_features'):
      dataset = tf.data.TFRecordDataset(input_shard_file_list)
      if element_count > 0:
        dataset = dataset.take(element_count)
      if skip_count > 0:
        dataset = dataset.skip(skip_count)
      if do_filter:
        dataset = dataset.filter(filter_fn)
      dataset = dataset.repeat(num_epochs)
      if do_shuffle:
        dataset = dataset.shuffle(buffer_size=queue_capacity)
      dataset = dataset.map(
          parse_fn, num_parallel_calls=num_parallelism).batch(batch_size)
      images, labels = dataset.make_one_shot_iterator().get_next()
      return images, labels

  return input_fn


def exported_model_input_signature():
  inputs = {INPUT_FEATURE_NAME: tf.placeholder(tf.string, shape=(None,))}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)
