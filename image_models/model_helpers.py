"""Define the model for inception v3 model."""
import tensorflow.contrib.slim as slim
import sys
sys.path.insert(0, '/media/haoweiliu/Data/models/research/slim')
from nets import nets_factory
from preprocessing import preprocessing_factory

import math
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


def get_optimizer(optimizer_name,
                  learning_rate,
                  momentum=0.9,
                  adam_beta1=0.9,
                  adam_beta2=0.999,
                  epsilon=0.001,
                  rmsprop_decay=0.9,
                  sync_replicas=False,
                  replicas_to_aggregate=None,
                  num_replicas=None,
                  variable_averages=None,
                  variables_to_average=None):
  """Return an optimizer by name."""
  if optimizer_name == 'sgd':
    tf.logging.info('Using SGD optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    tf.logging.info('Using Momentum optimizer')
    optimizer = tf.train.tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
  elif optimizer_name == 'rms':
    tf.logging.info('Using RMSPropOptimizer')
    optimizer = tf.RMSPropOptimizer(
        learning_rate, rmsprop_decay, momentum=momentum, epsilon=epsilon)
  else:
    assert False, 'Unknown optimizer: %s' % optimizer_name

  if sync_replicas:
    assert replicas_to_aggregate is not None
    assert num_replicas is not None

    ### To use SyncReplicasOptimizer with an Estimator, you need to send sync_replicas_hook while calling the fit.
    ### https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
    optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate,
                                         num_replicas, variable_averages,
                                         variables_to_average)

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

def regularization_loss(enable_regularization=True):
  total_reg_loss = None
  if enable_regularization:
     reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
     if reg_losses:
        total_reg_loss = tf.add_n(reg_losses)
  return total_reg_loss

def log_loss_into_summary(primary_loss, reg_loss, total_loss):
  tf.summary.scalar('losses/primary', primary_loss)
  if reg_loss is not None:
     tf.summary.scalar('losses/regularization', reg_loss)
  tf.summary.scalar('losses/total loss', total_loss)

def get_total_loss(loss_fn, labels, logits, enable_regularization=True):
  losses = []
  primary_loss = loss_fn(labels, logits)
  losses.append(primary_loss)

  reg_loss = regularization_loss(enable_regularization)
  if reg_loss is not None:
    losses.append(reg_loss)

  total_loss = tf.add_n(losses)
  log_loss_into_summary(primary_loss, reg_loss, total_loss)
  return total_loss


def build_warmup_learning_rate(initial_lr,
                               total_steps,
                               current_global_step,
                               warmup_steps_fraction = 0.0):
    warmup_steps = int(total_steps * warmup_steps_fraction)
    tf.logging.info('warmup_steps: %d' % warmup_steps)
    warmup_lr = (
      initial_lr * tf.cast(current_global_step, tf.float32) / tf.cast(
          warmup_steps, tf.float32))
    return warmup_lr, warmup_steps


def build_learning_rate(global_step,
                        initial_lr,
                        lr_decay_type,
                        decay_factor,
                        total_steps,
                        decay_steps,
                        warmup_steps_fraction=0.0):
  if lr_decay_type == 'exponential':
    assert decay_factor is not None
    assert decay_steps is not None
    # lr = initial_lr * decay_factor ^ (global step/decay_step)
    lr = tf.train.exponential_decay(initial_lr,
                                    global_step,
                                    decay_steps,
                                    decay_factor,
                                    staircase=True)
  elif lr_decay_type == 'cosine':
    lr = 0.5 * initial_lr * (
        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
    #lr = tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps)
    #lr = tf.cast(global_step, tf.float32) / total_steps
  else:
    assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

  warmup_lr, warmup_steps = build_warmup_learning_rate(initial_lr,
                                         total_steps, global_step,
                                         warmup_steps_fraction)
  lr = tf.cond(global_step < warmup_steps, lambda: tf.minimum(warmup_lr, lr),
               lambda: lr)
  tf.summary.scalar('learning_rate', lr)
  return lr

def build_learning_rate_and_optimzier(
                      global_step,
                      total_steps,
                      optimizer_name = 'rms',
                      initial_learning_rate = 0.0005,
                      lr_decay_type = 'exponential',
                      decay_steps = 10000,
                      learning_rate_decay_factor=0.94,
                      moving_average_decay=0.9999,
                      variables_to_optimize = tf.trainable_variables()
                      ):
    lr = build_learning_rate(global_step,
                             total_steps,
                             initial_learning_rate,
                             lr_decay_type,
                             decay_steps,
                             learning_rate_decay_factor)

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)

    return get_optimizer(optimizer_name, lr,
                              variable_averages=variable_averages,
                              variables_to_average=variables_to_optimize,
                              sync_replicas=True,
                              replicas_to_aggregate=1,
                              num_replicas=1)



def get_model_fn(total_steps,
                 num_categories,
                 input_processor,
                 learning_rate=0.001,
                 retrain_model=False,
                 dropout_rate=0.2,
                 optimizer_to_use = 'rms'):
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
      # (?, 299, 299, 3)
      images = tf.map_fn(processing_model_input, features, dtype=tf.float32)

    # (?, 2048)
    #feature_vector = inception_v3_module(images)
    arg_scope = inception.inception_v3_arg_scope(
        weight_decay=0.1, activation_fn=tf.nn.relu6)
    with tf.contrib.slim.arg_scope(arg_scope):
      features, end_points  = inception.inception_v3(
          images,
          num_classes=None,
          is_training=True,
          dropout_keep_prob=0.9)
    #features: (:, 1, 1, 2048)
    print(features)
    exit()
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
    ## add label smoothing
    loss = get_total_loss(tf.losses.sparse_softmax_cross_entropy, labels, logits)
    if is_training_mode:
      ## when we are not in fine tuning mode, remove the inceptionv3 variables
      optimizer = build_learning_rate_and_optimzier(tf.train.get_global_step(),
                                        total_steps,
                                        optimizer_name=optimizer_to_use,
                                        initial_learning_rate=learning_rate)
      sync_replicas_hook = optimizer.make_session_run_hook(True)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
      logging_hook = CreateLogger(tf.trainable_variables())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                        training_hooks=[logging_hook, sync_replicas_hook])

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

def get_raw_model_fn_with_pretrained_model(
                 total_steps,
                 num_categories,
                 input_processor,
                 learning_rate=0.001,
                 retrain_model=False,
                 dropout_rate=0.2,
                 optimizer_to_use = 'rms'):
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
    loss = get_total_loss(tf.losses.sparse_softmax_cross_entropy, labels, logits)
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




