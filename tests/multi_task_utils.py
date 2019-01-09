"""A base line model for us to verify/test multi-task training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import visual_shopping_image_utils as image_utils

FEATURE_KEY = 'feature/floats'


def create_model_fn(create_estimator_spec_func, task_name_to_kernels,
                    task_name_to_num_classes, no_optimizer=False):
  """Creates a model_fn for tensorflow Estimator.

  Args:
    create_estimator_spec_func: A function that creates the estimator spec.
    task_name_to_kernels: A dictionary of task name to logit kernels. We make it
      as a parameter to make it easier to test.
    task_name_to_num_classes: A dictionary of task name to number of classes for
      each task.
    no_optimizer: When True, gradient descent is disabled for testing purpose.

  Returns:
      A model_fn to be used to create an Estimator.
  """

  def multi_task_model_fn(features, labels, mode):
    """A multi-task model_fn for tensorflow Estimator.

    Note: The model_fn is a callback and will be invoked by tensorflow. The
      input variables must have features, labels, and mode.

    Args:
      features: Feature input for the classifier.
      labels: A dictionary of task name to ground truth label tensor
        of shape [batch_size, 1].
      mode: Estimator mode: TRAINING, EVAL or PREDICT.

    Returns:
        Am Estimator spec.
    """
    # Creates a logit for each task. We don't do any feature processing here for
    # simplicity but there could be a feature processor (e.g. inception3
    # embedding) unit prior to logit creation.
    task_name_to_logits = {}
    for task_name, task_classes in task_name_to_num_classes.items():
      task_name_to_logits[task_name] = tf.layers.dense(
          features[FEATURE_KEY],
          units=task_classes,
          kernel_initializer=tf.constant_initializer(
              task_name_to_kernels[task_name], verify_shape=True),
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
          name=task_name + '_logit',
          bias_initializer=tf.zeros_initializer)

    # Sets up the regularization loss for each task.
    task_name_to_reg_losses = {}
    for task_name in task_name_to_num_classes.keys():
      # The get_regularization_losses returns a list of regularization losses.
      # Since currently we only create a layer for each task, there will only
      # be one element in the list.
      # Adds an identity operation so we can have a easy-to-read name for the
      # regularization loss tensor.
      task_name_to_reg_losses[task_name] = tf.identity(tf.losses.get_regularization_losses(scope=task_name + '_logit')[0], name=task_name+'_reg_loss');

    return create_estimator_spec_func(task_name_to_logits, labels,
                                      task_name_to_num_classes, task_name_to_reg_losses, mode, no_optimizer)

  return multi_task_model_fn


def multi_task_estimator_spec_fn(task_name_to_logits, task_name_to_labels,
                                 task_name_to_num_classes,
                                 task_name_to_reg_losses, mode, no_optimizer):
  """A function that creates an Estimator spec.

  Args:
    task_name_to_logits: A dictionary of task name to logit tensor of shape, [batch_size,
      num_class]. Each row is an unnormalized probability of the corresponding
      example belonging to each class.
    task_name_to_labels: A dictionary of task name to ground truth label tensor of shape,
      [batch_size, 1].
    task_name_to_num_classes: A dictionary of task name to number of classes
      for the task.
    task_name_to_reg_losses: A dictionary of task name to task regularization
      loss.
    mode: Estimator mode: TRAINING, EVAL or PREDICT.
    no_optimizer: When True, gradient descent is disabled for testing purpose.

  Returns:
        Am Estimator spec.
  """
  task_names = task_name_to_num_classes.keys()
  if mode == tf.estimator.ModeKeys.PREDICT:
    prediction_result = {}
    export_outputs = {}
    for task_name in task_names:
      # Sets up prediction and exports output.
      probability = tf.nn.softmax(task_name_to_logits[task_name], name='probabilities')
      prediction_result[task_name + '/probabilities'] = probability
      prediction_result[task_name + '/top_class'] = tf.nn.top_k(
          probability, k=1).indices
      export_outputs[
          task_name +
          '/classification'] = tf.estimator.export.ClassificationOutput(
              scores=probability)
    # For multi-head models, tensorflow requires a default output. Here we
    # default it to the output of first task.
    export_outputs[tf.saved_model.signature_constants
                   .DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs[
                       task_names[0] + '/classification']

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=prediction_result, export_outputs=export_outputs)

  # Creates loss tensor.
  total_loss = 0
  for task_name, task_classes in task_name_to_num_classes.items():
    total_loss += image_utils.build_task_loss(
        labels=task_name_to_labels[task_name],
        logits=task_name_to_logits[task_name],
        task_name=task_name,
        task_classes=task_classes,
        label_smoothing=0.0)

  total_reg_loss = tf.identity(image_utils.build_weighted_task_reg_loss(task_name_to_reg_loss=task_name_to_reg_losses, task_name_to_labels=task_name_to_labels), name='total_reg_loss')
  total_loss += total_reg_loss
  loss_tensor = tf.identity(total_loss, name='total_loss')

  if mode == tf.estimator.ModeKeys.EVAL:
    multi_task_metric_func = image_utils.build_multi_task_metric_func(
        dataset_split_name='validation')
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss_tensor,
        eval_metric_ops=multi_task_metric_func(task_name_to_labels, task_name_to_logits))

  if mode == tf.estimator.ModeKeys.TRAIN:
    if no_optimizer:
      train_op = tf.no_op()
    else:
      train_op = tf.contrib.training.create_train_op(
          loss_tensor, tf.train.GradientDescentOptimizer(0.01))
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss_tensor, train_op=train_op)
