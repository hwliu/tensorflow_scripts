import tensorflow as tf
from example_parser import TFExampleParser

def filter_empty_feature(image, _):
  raw_bytes = tf.decode_raw(image[INPUT_FEATURE_NAME], tf.uint8)
  return tf.greater(tf.size(raw_bytes), 0)


def parser(value, dataset_parser):
  """Parses the raw data from SSTable into image and labels."""
  images, labels = dataset_parser.parse_single_example(value)
  image = {INPUT_FEATURE_NAME: images}
  label = convert_label(labels)
  return image, label

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

def create_input_fn_for_images_sstable(input_file_names,
                                       keys_to_features_map,
                                       dataset_type,
                                       mode,
                                       batch_size=50,
                                       do_shuffle=True,
                                       do_filter=True,
                                       num_parallelism=128,
                                       queue_capacity=256,
                                       element_count=-1,
                                       skip_count=0):
  """Creates and returns an input_fn based on input parameters."""
  dataset_parser = create_dataset_parser(dataset_type, keys_to_features_map)

  def input_fn():
    """Creates an input_fn based on input parameters."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    num_epochs = None if is_training else 1

    parse_fn = (lambda x: parser(x, dataset_parser))

    input_shard_file_list = shardedfile_to_filelist(input_file_names)
    with tf.name_scope('read_batch_features'):
      dataset = tf.data.TFRecordDataset(input_shard_file_list)
      if element_count > 0:
        dataset = dataset.take(element_count)
      if skip_count > 0:
        dataset = dataset.skip(skip_count)
      if do_shuffle:
        dataset = dataset.shuffle(buffer_size=queue_capacity)

      dataset = dataset.map(parse_fn, num_parallel_calls=num_parallelism)
      if do_filter:
        dataset = dataset.filter(filter_empty_feature)

      dataset = dataset.repeat(num_epochs).batch(batch_size)
      dataset = dataset.prefetch(1024)
      images, labels = dataset.make_one_shot_iterator().get_next()
      return images, labels

  return input_fn
