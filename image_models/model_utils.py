import tensorflow as tf

import tensorflow.contrib.slim as slim
import sys
sys.path.insert(0, '/media/haoweiliu/Data/models/research/slim')
from nets import nets_factory
from preprocessing import preprocessing_factory
import functools
import numpy as np

def select_network(network_name, num_classes, is_training):
  network_fn = nets_factory.get_network_fn(
      network_name,
      num_classes=num_classes,
      weight_decay=0.00004,
      is_training=is_training)
  network_fn.default_image_size
  return network_fn


def select_preprocessing_fn(network_name, is_training):
  def preprocessing_fn_with_option(image, output_height, output_width, channels,
                                   preprocess_options, **kwargs):
    del channels
    del preprocess_options
    return preprocessing_factory.get_preprocessing(
        network_name, is_training)(
            image=image,
            output_height=output_height,
            output_width=output_width,
            **kwargs)
  return preprocessing_fn_with_option


def build_aux_loss(loss_fn, end_points, labels,
                   loss_name='aux_loss'):
   aux_endpoint = end_points.get('aux_logits')
   if aux_endpoint is not None:
     aux_loss = loss_fn(
       labels,
       tf.squeeze(aux_endpoint, axis=[0]),
       weights=0.4,
       scope=loss_name
     )
     tf.summary.scalar('losses/' + loss_name, aux_loss)
     tf.add_loss(aux_loss)


def build_total_loss_function(loss_fn, loss_name,
                        logits, end_points, labels,
                        l2_weight_decay):
    build_aux_loss(loss_fn, end_points, labels)
    reg_loss = tf.losses.get_regularization_loss()
    tf.summary.scalar('losses/regularization_loss', reg_loss)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)
    return total_loss

def create_loss_function_with_class_num(is_multiclass,
                                        logits,
                                        end_points,
                                        labels,
                                        label_smoothing=0.1,
                                        l2_weight_decay=None):
    loss_fn = tf.losses.softmax_cross_entropy
    loss_name = 'softmax_loss'
    if is_multiclass:
      loss_fn = tf.losses.sigmoid_cross_entropy
      loss_name = 'sigmoid_loss'
    loss_fn_with_options = functools.partial(loss_fn, label_smoothing=label_smoothing)
    return build_total_loss_function(
      loss_fn_with_options,
      loss_name,
      logits=logits,
      end_points=end_points,
      labels=labels,
      l2_weight_decay=l2_weight_decay
    )

def get_variables_to_average():
  variables_to_average = tf.trainable_variables()
  # Remove integer variables
  # Some older model (GNet) use global_step as integer variable.
  variables_to_average = [
      v for v in variables_to_average if v.dtype != 'int32_ref'
  ]
  variables_to_average = list(set(variables_to_average))
  return variables_to_average

def create_ema_function_and_variables(moving_average_decay, current_global_step):
  ema = tf.train.ExponentialMovingAverage(moving_average_decay,
                                          current_global_step)
  variables_to_average = get_variables_to_average()
  return ema, variables_to_average

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


def build_exponential_learning_rate(initial_lr,
                                    global_step,
                                    decay_steps,
                                    decay_factor):
    return tf.train.exponential_decay(initial_lr,
                                      global_step,
                                      decay_steps,
                                      decay_factor,
                                      staircase=True)


def build_cosine_learning_rate(initial_lr,
                               global_step,
                               total_steps):
    return 0.5 * initial_lr * (
        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))

def build_learning_rate(global_step,
                        initial_lr,
                        lr_decay_type,
                        decay_factor,
                        decay_steps,
                        total_steps,
                        replicas_to_aggregate,
                        warmup_steps_fraction=0.0):
  if replicas_to_aggregate > 1:
    initial_lr *= replicas_to_aggregate

  if lr_decay_type == 'exponential':
    lr = build_exponential_learning_rate(initial_lr,
                                         global_step,
                                         decay_steps,
                                         decay_factor)
  elif lr_decay_type == 'cosine':
    lr = build_cosine_learning_rate(initial_lr, global_step, total_steps)
  else:
    assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

  warmup_lr, warmup_steps = build_warmup_learning_rate(initial_lr,
                                         total_steps, global_step,
                                         warmup_steps_fraction)
  lr = tf.cond(global_step < warmup_steps, lambda: tf.minimum(warmup_lr, lr),
               lambda: lr)
  tf.summary.scalar('learning_rate', lr)
  return lr


def create_optimizer(optimizer_name,
                  learning_rate,
                  momentum=0.9,
                  adam_beta1=0.9,
                  adam_beta2=0.999,
                  epsilon=0.001,
                  rmsprop_decay=0.9,
                  replicas_to_aggregate=None,
                  ema_average_func=None,
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

  if replicas_to_aggregate > 1:
    ### To use SyncReplicasOptimizer with an Estimator, you need to send sync_replicas_hook while calling the fit.
    ### https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
    optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate,
                                         replicas_to_aggregate, ema_average_func,
                                         variables_to_average)

  return optimizer


def get_probabilities_and_labels_from_logits(logits):
  probs = tf.nn.softmax(logits, name='softmax_tensor')
  predicted_labels = tf.argmax(input=probs, axis=1)
  return predicted_labels, probs


def metric_fn(ground_truth_labels, predicted_labels, predicted_probs):
  return {
      'accuracy': tf.metrics.accuracy(labels=ground_truth_labels,
                                       predictions=predicted_labels),
      'auc': tf.metrics.auc(ground_truth_labels, predicted_probs),
      'auc_pr': tf.metrics.auc(ground_truth_labels, predicted_probs,
                               curve='PR')
  }


def proprocess_input(img_encoded, preprocess_fn, default_image_size):
  image = tf.image.decode_image(img_encoded, 3)
  image = preprocess_fn(image, default_image_size)
  return image


def create_model_fn(network_fn,
                    training_preprocessing_fn,
                    eval_preprocessing_fn,
                    num_classes,
                    is_multilabel,
                    num_of_replicas=1,
                    label_smoothing=0.1,
                    moving_average_decay=0.9999,
                    decay_steps=100000,
                    total_steps=300000,
                    learning_rate=0.01,
                    leraning_rate_decay_type='cosine',
                    optimizer_to_use='rmsprop'):

   def model_fn(feature, labels, mode):
      # feature: a string
      image_preprocessing_fn = training_preprocessing_fn if mode == tf.estimator.ModeKeys.TRAIN else eval_preprocessing_fn
      preprocess_fn =  (lambda x: proprocess_input(x, image_preprocessing_fn, network_fn.image_size))
      images = tf.map_fn(preprocess_fn, feature, tf.float32)
      logits, end_points = network_fn(images)
      losses = create_loss_function_with_class_num(is_multilabel,
                                          logits,
                                          end_points,
                                          labels,
                                          label_smoothing=label_smoothing)
      global_step = tf.train.get_or_create_global_step()
      ema, variables_to_average = create_ema_function_and_variables(
                                  moving_average_decay=moving_average_decay,
                                  current_global_step=global_step)
      lr = build_learning_rate(global_step,
                               learning_rate,
                               leraning_rate_decay_type,
                               decay_factor=0.97,
                               decay_steps=decay_steps,
                               total_steps=total_steps,
                               replicas_to_aggregate=num_of_replicas)
      optimizer = create_optimizer(optimizer_to_use, lr,
                                  replicas_to_aggregate=num_of_replicas,
                                  ema_average_func=ema,
                                  variables_to_average=variables_to_average)
      predicted_labels, probs = get_probabilities_and_labels_from_logits(logits)
      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          'classes': predicted_labels,
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          'scores': probs
      }
      # For prediction mode
      export_outputs = {
          'classify': tf.estimator.export.PredictOutput(predictions)
      }

      sync_replicas_hook = optimizer.make_session_run_hook(True)
      train_op = optimizer.minimize(
          loss=losses, global_step=tf.train.get_global_step())
      metrics_ops = metric_fn(labels, predicted_labels, probs)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=losses, train_op=train_op,
                                        training_hooks=[sync_replicas_hook])
      elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=predictions,
                                        export_outputs=export_outputs)
      else:
        return tf.estimator.EstimatorSpec(mode=mode, loss=losses,
                                        eval_metric_ops=metrics_ops)
