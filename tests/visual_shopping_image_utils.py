"""Tensorflow utility functions for an image training pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def build_total_loss(logits,
                     onehot_labels,
                     label_smoothing=0.1,
                     add_regularization_losses=True,
                     add_summary=True,
                     scope_name='',
                     weights=1.0):
  """Builds the total loss tensor.

  Args:
      logits: A tensor of shape [num_examples, num_classes] and of type float,
        that represents the unnormalized probability of each example belonging
        to each class.
      onehot_labels: An indicator tensor of shape [num_examples, num_classes]
        and of type int32. The vector at row i is the one-hot label for example
        i.
      label_smoothing: A float parameter for label smoothing.
      add_regularization_losses: Whether or nor to add regularization loss.
      add_summary: Whether or not to log the loss information. Please set to
        False when running with TPU.
      scope_name: Prefix for the logging name for loss.
      weights: A tensor that has the shape, [batch_size, 1] indicating example
        weights.

  Returns:
      The tensor that holds the total loss.
  """
  primary_loss = tf.losses.softmax_cross_entropy(
      onehot_labels,
      logits,
      label_smoothing=label_smoothing,
      weights=weights,
      scope=scope_name + 'softmax_loss')
  total_loss = primary_loss

  reg_loss = tf.losses.get_regularization_loss()
  if add_regularization_losses:
    total_loss += reg_loss
  if add_summary:
    tf.summary.scalar('{}losses/primary_loss'.format(scope_name), primary_loss)
    tf.summary.scalar('{}losses/total_loss'.format(scope_name), total_loss)
    if add_regularization_losses:
      tf.summary.scalar('{}losses/reg_loss'.format(scope_name), reg_loss)

  return total_loss


def build_weight_for_label(labels):
  """Builds the weight tensor given a label tensor.

  Args:
      labels: A tensor of shape [num_examples, 1] and of type int32 or int64,
        that represents the class labels.

  Returns:
      A weight tensor of either 0 or 1 for each element. 0-elements corresponds
      to those equal to -1 in labels.
  """
  # Creates a weight tensor and sets the weights to 0 if the corresponding
  # element in labels is negative: first creates an all-one vector of the same
  # size as labels, then creates a mask where the elements having negative
  # values in labels has 0 as mask value. Multiplying the two tensors gives us
  # the proper weight tensor.
  return tf.multiply(
      tf.ones_like(labels),
      tf.cast(tf.greater_equal(labels, tf.zeros_like(labels)), labels.dtype))


def build_task_loss(labels,
                    logits,
                    task_name,
                    task_classes,
                    label_smoothing=0.0):
  """Builds the loss function for individual task for multi-task training.

  Note:
      regularization loss is not included.

  Args:
      labels: A tensor of shape [batch_size, 1] and of type int32 or int64, that
        represents the class labels.
      logits: A tensor of shape [batch_size, num_classes] and of type float,
        that represents the unnormalized probability of each example belonging
        to each class.
      task_name: Name of the task, for logging purpose.
      task_classes: Number of classes of the task.
      label_smoothing: A float parameter for label smoothing.

  Returns:
      The tensor that holds the total loss.
  """
  return build_total_loss(
      logits=logits,
      onehot_labels=tf.one_hot(labels, task_classes),
      add_regularization_losses=False,
      scope_name=task_name + '/',
      label_smoothing=label_smoothing,
      weights=build_weight_for_label(labels))


def build_learning_rate(initial_learning_rate,
                        global_step,
                        decay_steps=250000,
                        add_summary=True,
                        warmup_steps=2500):
  """Builds a learning rate schedule."""
  learning_rate = tf.train.cosine_decay(initial_learning_rate, global_step,
                                        decay_steps)
  tf.logging.info('Warmup_steps: {}'.format(warmup_steps))
  warmup_learning_rate = (
      initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
          warmup_steps, tf.float32))

  learning_rate = tf.cond(
      global_step < warmup_steps,
      lambda: tf.minimum(warmup_learning_rate, learning_rate),
      lambda: learning_rate)

  if add_summary:
    tf.summary.scalar('learning_rate', learning_rate)

  return learning_rate


def build_optimizer(name,
                    learning_rate,
                    momentum=0.9,
                    epsilon=0.001,
                    rmsprop_decay=0.9,
                    sync_replicas=False,
                    num_replicas=1,
                    moving_average=None,
                    variables_to_average=None):
  """Builds an optimizer."""
  if name == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum, use_nesterov=True)
  elif name == 'rmsprop':
    optimizer = tf.RMSPropOptimizer(
        learning_rate, rmsprop_decay, momentum=momentum, epsilon=epsilon)
  else:
    raise Exception('Unknown optimizer : {}'.format(name))

  # If multiple workers are used for training, average the gradients from the
  # workers before updating.
  if sync_replicas:
    optimizer = tf.train.SyncReplicasOptimizer(optimizer, num_replicas,
                                               num_replicas, moving_average,
                                               variables_to_average)
  return optimizer


def build_multi_task_metric_func(dataset_split_name, add_summary=True):
  """Builds a metric_fn for multi-task evaluation.

  Args:
    dataset_split_name: Name of the dataset split for evaluation, e.g.
      validation or test. This is only for logging purpose.
    add_summary: Whether should we dump the results to tensor board.

  Returns:
    A metric function that can be used to generate metric operations.
  """

  def multi_task_metric_func(labels, logits):
    """Generates metrics operations for multi-task evaluation.

    Args:
      labels: A dictionary of task name to label tensors of shape [num_examples,
        1].
      logits: A dictionary of task name to logit tensors of shape [num_examples,
        num_classes].

    Raises:
      ValueError if labels and logits do not have the same keys.

    Returns:
        A dictionary of key to metrics operation.
    """
    if labels.keys() != logits.keys():
      raise ValueError('Task names are different for labels and logits.')
    metric_ops = {}
    for task_name, label in labels.items():
      accuracy_metric_name = '{}/Eval/Accuracy/{}'.format(
          task_name, dataset_split_name)
      metric_ops[accuracy_metric_name] = tf.metrics.accuracy(
          label,
          tf.argmax(logits[task_name], 1),
          weights=build_weight_for_label(label))

    if add_summary:
      for name, value in metric_ops.items():
        tf.summary.scalar(name, value)
    return metric_ops

  return multi_task_metric_func


def build_metric_func(dataset_split_name, add_summary=True):
  """Gets a metric_fn and names it with dataset_split_name."""

  def metric_func(labels, logits):
    """Evaluation metric function that runs on CPU."""
    accuracy_metric_name = 'Eval/Accuracy/%s' % dataset_split_name
    metric_map = {
        accuracy_metric_name: tf.metrics.accuracy(labels, tf.argmax(logits, 1)),
    }
    if add_summary:
      for name, value in metric_map.items():
        tf.summary.scalar(name, value)
    return metric_map

  return metric_func


def get_variables_to_restore_from_pretrain_checkpoint(exclude_scopes,
                                                      variable_shape_map=None):
  """Gets variable names to be restored from pretrained checkpoint.

  The function checks all the trainable variables against the variable shape
  information, loaded from the checkpoint and excludes those starting with names
  in exclude_scopes. global_step is also excluded.

  Args:
      exclude_scopes: Comma-separated scopes of the variables to exclude.
      variable_shape_map: The map of the variables and their shapes in the
        checkpoint.

  Returns:
      The names of the variables to be loaded from the checkpoint.
  """
  # Skips restoring global_step.
  exclusions = ['global_step']
  if exclude_scopes:
    exclusions.extend([scope.strip() for scope in exclude_scopes.split(',')])

  variable_to_restore = {
      # Uses get_model_variables to get all the model variables. These include
      # those from tf.trainable_variables() and batch norm variables, such as
      # moving variance and moving mean.
      v.op.name: v for v in tf.contrib.framework.get_model_variables()
  }
  # Removes variables from exclude_scope.
  filtered_variables_to_restore = {}
  for variable_name, tensor in variable_to_restore.items():
    excluded = False
    for exclusion in exclusions:
      if variable_name.startswith(exclusion):
        excluded = True
        tf.logging.info('Exclude var {}'.format(variable_name))
        break
    if not excluded:
      filtered_variables_to_restore[variable_name] = tensor

  # Removes variables that have incompatible shape or not in the checkpoint.
  final_variables_to_restore = {}
  for variable_name, tensor in filtered_variables_to_restore.items():
    if variable_name not in variable_shape_map:
      tf.logging.info(
          'Skip var {} because it is not in map.'.format(variable_name))
      continue

    if not tensor.get_shape().is_compatible_with(
        variable_shape_map[variable_name]):
      tf.logging.info(
          'Skip init [%s] from [%s] in ckpt because shape mismatch: %s vs %s',
          tensor.name, variable_name, tensor.get_shape(),
          variable_shape_map[variable_name])
      continue

    final_variables_to_restore[variable_name] = tensor

  for variable_name, tensor in final_variables_to_restore.items():
    tf.logging.info('Init variable [%s] from [%s] in ckpt', variable_name,
                    tensor)

  return final_variables_to_restore


def _init_checkpoint_and_variables(pretrain_checkpoint_path,
                                   pretrain_checkpoint_exclude_scopes):
  """Retrieves the variables to be loaded from a checkpoint.

  Args:
      pretrain_checkpoint_path: Path to the checkpoint file.
      pretrain_checkpoint_exclude_scopes: Comma-separated scopes of the
        variables to exclude.

  Raises:
      ValueError: If the checkpoint path is None.
      tf.errors.NotFoundErrors: If the checkpoint path doesn't exist.

  Returns:
      The variables to be loaded from the checkpoint.
  """
  checkpoint_reader = tf.contrib.framework.load_checkpoint(
      pretrain_checkpoint_path)
  return get_variables_to_restore_from_pretrain_checkpoint(
      pretrain_checkpoint_exclude_scopes,
      checkpoint_reader.get_variable_to_shape_map())


def build_pretrained_loader_fn(pretrain_checkpoint_path,
                               checkpoint_exclude_scopes):
  """Returns a scaffold object to load variables from a pretrained checkpoint.

  Args:
      pretrain_checkpoint_path: Path to the checkpoint file.
      checkpoint_exclude_scopes: Scopes of the variables to exclude.

  Returns:
      A scaffolding object to load variables from a checkpoint.
  """
  scaffolding = None
  try:
    tf.logging.info(
        'Attempts to load checkpoint: {}'.format(pretrain_checkpoint_path))
    var_list = _init_checkpoint_and_variables(
        pretrain_checkpoint_path=pretrain_checkpoint_path,
        pretrain_checkpoint_exclude_scopes=checkpoint_exclude_scopes)
  except (tf.errors.NotFoundError, ValueError) as error:
    tf.logging.error(error)
    var_list = None

  if var_list is not None:
    assign_fn = (
        tf.contrib.framework.assign_from_checkpoint_fn(
            pretrain_checkpoint_path, var_list, ignore_missing_vars=True))
    init_fn = lambda _, session: assign_fn(session)
    scaffolding = tf.train.Scaffold(init_fn=init_fn)

  return scaffolding
