"""Define the model for inception v3 model."""
import sys
sys.path.insert(0, '/media/haoweiliu/Data/models/research/slim')
import numpy as np
import re
import tensorflow as tf
import tensorflow_hub as hub
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INCEPTION_V3_MODULE_PATH
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INCEPTION_V3_TARGET_IMAGE_SIZE
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INPUT_FEATURE_NAME
#from google3.experimental.users.haoweiliu.watermark_training.input_processing_helpers import preprocess_image
import tensorflow.contrib.slim as slim
from nets import inception
from inception_v3_on_watermark_settings import INCEPTION_V3_MODULE_PATH
from inception_v3_on_watermark_settings import INCEPTION_V3_TARGET_IMAGE_SIZE
from inception_v3_on_watermark_settings import INPUT_FEATURE_NAME
from input_preprocessing_helpers import preprocess_image
from logger_hook import CreateLogger

_INCEPTION_V3_ORIGINAL_CHECKPOINT = '/media/haoweiliu/Data/tensorflow_scripts/dataset/inception_v3.ckpt'
TEMP_CHECKPOINT_PATH = '/tmp/new_checkpoint.ckpt'

def _get_first_layer_variable_reg_expression(feature_extractor_name):
  """Returns the regular expression for the first layer conv variable."""
  feature_extractor_name = feature_extractor_name.lower()
  if feature_extractor_name == 'mobilenet_v1':
    return '(FeatureExtractor/)?Mobilenet[^/]*/Conv2d_0/weights'
  elif feature_extractor_name == 'mobilenet_v2':
    return '(FeatureExtractor/)?Mobilenet[^/]*/Conv/weights'
  elif feature_extractor_name == 'resnet_v1':
    return '(FeatureExtractor/)?resnet_v1[^/]*/conv1/weights'
  elif feature_extractor_name == 'resnet_v2':
    return '(FeatureExtractor/)?resnet_v2[^/]*/conv1/weights'
  elif feature_extractor_name == 'inception_v2':
    return '(FeatureExtractor/)?InceptionV2/Conv2d_1a_7x7/depthwise_weights'
  elif feature_extractor_name == 'inception_v3':
    return '(FeatureExtractor/)?InceptionV3/Conv2d_1a_3x3/weights'
  elif feature_extractor_name == 'inception_resnet_v2':
    return '(FeatureExtractor/)?InceptionResnetV2/Conv2d_1a_3x3/weights'
  else:
    raise ValueError('Did not recognize the feature extractor name.')

def convert_checkpoint_to_custom_channel_input(
    checkpoint_path,
    new_checkpoint_path,
    num_input_channels,
    feature_extractor_name):
  """Converts a checkpoint to accommodate inputs with arbitrary channels.

  Given the original checkpoint, this function will modify the first
  convolutional layer so that it can operate on inputs with `num_input_channels`
  depth. Specifically, it will tile (or slice) the original checkpoint variable
  weight to occupy the new channel depth.

  Args:
    checkpoint_path: Path to the original detection checkpoint.
    new_checkpoint_path: Path to write converted checkpoint.
    num_input_channels: The number of input channels that the new model will
      use, as an int32.
    feature_extractor_name: Name of the feature extractor. Options are:
      'mobilenet_v1'
      'mobilenet_v2'
      'resnet_v1'
      'resnet_v2'
      'inception_v2',
      'inception_v3',
      'inception_resnet_v2'

  Raises:
    RuntimeError: If the expected first layer variable is not found in the
      provided checkpoint.
  """
  with tf.Graph().as_default():
    first_layer_reg_expr = _get_first_layer_variable_reg_expression(
        feature_extractor_name)
    p = re.compile(first_layer_reg_expr)
    reader = tf.train.load_checkpoint(checkpoint_path)
    name_to_shape_map = reader.get_variable_to_shape_map()
    name_to_variable_map = {}
    found_first_layer_variable = False
    for name in name_to_shape_map:
      value = reader.get_tensor(name)
      if p.match(name):
        found_first_layer_variable = True
        value = _convert_weight_by_tiling_or_slice(value,
                                                   num_input_channels)
      name_to_variable_map[name] = tf.Variable(value)
    if not found_first_layer_variable:
      raise RuntimeError('Did not find variable matching regular expression '
                         '{0} in checkpoint {1}'.format(
                             first_layer_reg_expr, checkpoint_path))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(name_to_variable_map)
    with tf.Session() as sess:
      sess.run(init)
      tf.logging.info('Writing modified checkpoint to {}'.format(
          new_checkpoint_path))
      saver.save(sess, new_checkpoint_path, write_meta_graph=False)


def _convert_weight_by_tiling_or_slice(value,
                                       num_input_channels,
                                       normalize_along_channels=True):
  """Converts a numpy array to a new channels dimension by tile or slice.

  Args:
    value: A [K1, K2, channels_in, channels_out] numpy array holding a conv
      weight. K1 and K2 are the spatial dimensions of the filter.
    num_input_channels: Desired number of input channels in the new checkpoint.
    normalize_along_channels: Whether each spatial position should be normalized
      (after tiling or slicing) such that the sum of weights along the channel
      dimension remains the same.

  Returns:
    A [K1, K2, new_channels_in, channels_out] numpy array with the new conv
    weight.
  """
  original_num_channels = value.shape[2]
  original_sum = np.sum(value, axis=2, keepdims=True)
  print('original channel: {}'.format(original_num_channels))
  if num_input_channels > original_num_channels:
    num_tiles = np.ceil(float(num_input_channels) / original_num_channels)
    value = np.tile(value, [1, 1, num_tiles.astype(np.int32), 1])
  weight = value[:, :, :num_input_channels, :]
  if normalize_along_channels:
    new_sum = np.sum(weight, axis=2, keepdims=True)
    weight *= original_sum / new_sum
  return weight


def get_optimizer(optimizer_name, learning_rate):
  """Return an optimizer by name."""
  if optimizer_name == 'sgd':
    tf.logging.info('Using SGD optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    tf.logging.info('Using Momentum optimizer')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9)
  elif optimizer_name == 'rms':
    tf.logging.info('Using RMSPropOptimizer')
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  else:
    tf.logging.fatal('Unknown optimizer:', optimizer_name)
    optimizer = None

  return optimizer


def processing_model_input(input_feature):
  image = preprocess_image(
      input_feature[INPUT_FEATURE_NAME],
      tf.constant(INCEPTION_V3_TARGET_IMAGE_SIZE, dtype=tf.int32))
  return image


def metric_fn(ground_truth_labels, predicted_labels, predicted_probs):
  return {
      'accuracy': tf.metrics.accuracy(labels=ground_truth_labels,
                                       predictions=predicted_labels),
      'auc': tf.metrics.auc(ground_truth_labels, predicted_probs),
      'auc_pr': tf.metrics.auc(ground_truth_labels, predicted_probs,
                               curve='PR')
  }


def add_final_layer(feature_vector,
                    num_classes,
                    *args, **kargs):
    return tf.layers.dense(inputs=feature_vector,
              units=num_classes,
              *args, **kargs)


def get_probabilities_and_labels_from_logits(logits):
  probs = tf.nn.softmax(logits, name='softmax_tensor')
  predicted_labels = tf.argmax(input=probs, axis=1)
  return predicted_labels, probs

def softmax_cross_entropy_loss(labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)


def get_model_fn(num_categories,
                 input_processor,
                 learning_rate=0.001,
                 retrain_model=False,
                 dropout_rate=0.2,
                 optimizer_to_use = 'sgd'):
  """Wrapper of the inception v3 model function."""
  def inception_v3_model_fn(features, labels, mode):
    """Model function for inception V3."""
    is_training_mode = (mode == tf.estimator.ModeKeys.TRAIN)
    is_predict_mode = (mode == tf.estimator.ModeKeys.PREDICT)
    # Load Inception-v3 model.
    inception_v3_module = hub.Module(
        INCEPTION_V3_MODULE_PATH, trainable=retrain_model)
    if input_processor is not None:
      images = input_processor(features)
    else:
      images = tf.map_fn(processing_model_input, features, dtype=tf.float32)

    feature_vector = inception_v3_module(images)
    logits = add_final_layer(feature_vector, num_categories, activation=tf.nn.softmax)
    predicted_labels, probs = get_probabilities_and_labels_from_logits(logits)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': predicted_labels,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'scores': probs
    }

    if is_predict_mode:
      export_outputs = {
          'classify': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = softmax_cross_entropy_loss(labels, logits)
    if is_training_mode:
      optimizer = get_optimizer(optimizer_to_use, learning_rate)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
      hook = CreateLogger(tf.trainable_variables())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                        training_hooks=[hook])

    metrics_ops = metric_fn(labels, predicted_labels, probs)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=metrics_ops)

  return inception_v3_model_fn


def remove_variables_from_list(variables_to_remove, variable_list):
  remaining_variables=[]
  for v in variable_list:
    if v not in variables_to_remove:
      remaining_variables.append(v)
  return remaining_variables

def get_raw_model_fn_with_pretrained_model(num_categories,
                     input_processor=None,
                     learning_rate=0.001,
                     retrain_model=False,
                     dropout_rate=0.2,
                     optimizer_to_use = 'sgd'):
  """Wrapper of the raw inception v3 model function."""
  checkpoint_path = TEMP_CHECKPOINT_PATH
  convert_checkpoint_to_custom_channel_input(
      _INCEPTION_V3_ORIGINAL_CHECKPOINT, checkpoint_path, 3, 'inception_v3')

  def inception_v3_model_fn(features, labels, mode):
    """Model function for inception V3."""
    is_training_mode = (mode == tf.estimator.ModeKeys.TRAIN)
    is_predict_mode = (mode == tf.estimator.ModeKeys.PREDICT)

    if input_processor is not None:
      images = input_processor(features)
    else:
      images = tf.map_fn(processing_model_input, features, dtype=tf.float32)

    images = tf.multiply(images, 2.0)
    images = tf.subtract(images, 1.0)
    with slim.arg_scope(
         inception.inception_v3_arg_scope()):
            feature_vector, _ = inception.inception_v3(
               images,
               num_classes=None,
               is_training=False)

    feature_vector = tf.squeeze(feature_vector, [1, 2], name='SpatialSqueeze')

    asg_map = {
        v.op.name: v.op.name
        for v in tf.global_variables()
        if v.name.startswith('InceptionV3/')
    }
    inception_v3_model_variables = []
    for v in tf.global_variables():
      if v.name.startswith('InceptionV3/'):
        inception_v3_model_variables.append(v)
    tf.contrib.framework.init_from_checkpoint(checkpoint_path, asg_map)

    logits = add_final_layer(feature_vector, num_categories, activation=tf.nn.softmax)
    predicted_labels, probs = get_probabilities_and_labels_from_logits(logits)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': predicted_labels,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'scores': probs
    }

    if is_predict_mode:
      export_outputs = {
          'classify': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = softmax_cross_entropy_loss(labels, logits)
    if is_training_mode:
      optimizer = get_optimizer(optimizer_to_use, learning_rate)
      variables_to_optimization = remove_variables_from_list(
          inception_v3_model_variables, tf.trainable_variables())
      train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            var_list=variables_to_optimization)
      hook = CreateLogger(variables_to_optimization)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                        train_op=train_op,
                                        training_hooks=[hook])

    metrics_ops = metric_fn(labels, predicted_labels, probs)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=metrics_ops)

  return inception_v3_model_fn
