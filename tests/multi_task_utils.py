"""Utility functions for setting up multi-task co-training training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import visual_shopping_image_utils as image_utils

FEATURE_KEY = 'feature/floats'


def create_model_fn(create_estimator_spec_fn):

  def model_fn(features, labels, mode):
    return create_estimator_spec_fn(features[FEATURE_KEY], labels, mode)

  return model_fn

def create_estimator_spec_fn(tasks, kernels):

  def estimator_spec_fn(hidden_features, labels, mode):
    # Creates a logit for each task and sums the loss.
    task_names = []
    logits = {}
    for task_name, task_classes in tasks.items():
      logits[task_name] = tf.layers.dense(
          hidden_features,
          units=task_classes,
          kernel_initializer=tf.constant_initializer(kernels[task_name], verify_shape=True),
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
          name=task_name + '_logit',
          bias_initializer=tf.zeros_initializer)
      task_names.append(task_name)

    if mode == tf.estimator.ModeKeys.PREDICT:
      prediction_result = {}
      export_outputs = {}
      for task_name in task_names:
        # Sets up prediction output.
        probability = tf.nn.softmax(logits[task_name], name='probabilities')
        prediction_result[task_name+'/probabilities'] = probability
        prediction_result[task_name+'/top_class'] = tf.nn.top_k(probability, k=1).indices
        # Sets up export output.
        export_outputs[task_name+'/classification'] = tf.estimator.export.ClassificationOutput(scores=probability)
      # For multi-head models, tensorflow requires a default output. Here we default it to the first task.
      export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs[task_names[0]+'/classification']

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=prediction_result,
          export_outputs=export_outputs)

    # Creates loss tensor.
    total_loss = 0
    for task_name, task_classes in tasks.items():
      total_loss += image_utils.build_task_loss(
          labels=labels[task_name],
          logits=logits[task_name],
          task_name=task_name,
          task_classes=task_classes,
          label_smoothing=0.0)

    total_loss += tf.cast(tf.losses.get_regularization_loss(), total_loss.dtype)
    # Adds an identity operation so we can have a easy-to-read name for the
    # total loss tensor.
    loss_tensor = tf.identity(total_loss, name='total_loss')

    if mode == tf.estimator.ModeKeys.EVAL:
      multi_task_metric_func = image_utils.build_multi_task_metric_func(dataset_split_name='validation')
      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss_tensor,
        eval_metric_ops=multi_task_metric_func(labels, logits))

    if mode == tf.estimator.ModeKeys.TRAIN:
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss_tensor, train_op=tf.no_op())

    return 0

  return estimator_spec_fn
