"""Tests for functions inside visual_shopping_image_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import tf_testing_utils
import visual_shopping_image_utils as image_utils


class VisualShoppingImageUtilsTest(tf.test.TestCase):

  def test_build_total_loss(self):
    labels = np.array([1, 0, 1, 0, 1, 1])
    logits = np.array([[0.4, 0.3], [0.55, 0.37], [0.2, 0.8], [0.95, 0.9],
                       [0.7, 0.8], [0.3, 0.5]])
    loss_tensor = image_utils.build_total_loss(
        tf.constant(logits, dtype=tf.float32),
        tf.one_hot(labels, depth=2),
        label_smoothing=0.0)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      self.assertAlmostEqual(
          loss_tensor.eval(),
          tf_testing_utils.softmax_cross_entropy_loss(logits, labels),
          places=5)

  def test_build_metric_func(self):
    ground_truth_labels = np.array([1, 0, 0, 1, 0, 1])
    logits = np.array([[0.4, 0.3], [0.55, 0.37], [0.2, 0.8], [0.95, 0.9],
                       [0.7, 0.8], [0.3, 0.5]])
    metric_func = image_utils.build_metric_func(
        dataset_split_name='validation_set')
    result = metric_func(ground_truth_labels, logits)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      # tf.metrics.* requires local variable initializer.
      session.run(tf.local_variables_initializer())
      self.assertAlmostEqual(
          result['Eval/Accuracy/validation_set'][1].eval(), 0.33333, places=5)

  def test_build_multi_task_metric_func(self):
    # We use -1 to indicate don't-care examples. In this test case, we make
    # a classifier that is correct for example 1 and 5 for task 1 and
    # example 4 for task 2.
    ground_truth_labels = {
        'task1': np.array([-1, 0, -1, 1, -1, 1]),
        'task2': np.array([1, -1, 0, -1, 0, -1])
    }
    logits = {
        'task1':
            np.array([[0.4, 0.3], [0.55, 0.37], [0.2, 0.8], [0.95, 0.9],
                      [0.7, 0.8], [0.3, 0.5]]),
        'task2':
            np.array([[0.4, 0.3], [0.55, 0.37], [0.2, 0.8], [0.95, 0.9],
                      [0.85, 0.8], [0.3, 0.5]])
    }
    metric_func = image_utils.build_multi_task_metric_func(
        dataset_split_name='validation_set')
    result = metric_func(ground_truth_labels, logits)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      # tf.metrics.* requires local variable initializer.
      session.run(tf.local_variables_initializer())
      self.assertAlmostEqual(
          result['task1/Eval/Accuracy/validation_set'][1].eval(),
          0.6666667,
          places=5)
      self.assertAlmostEqual(
          result['task2/Eval/Accuracy/validation_set'][1].eval(),
          0.3333333,
          places=5)

  def test_restore_variables(self):
    tensor1 = tf.Variable(
        initial_value=0,
        name='test/v1',
        collections=[tf.GraphKeys.MODEL_VARIABLES])
    tensor2 = tf.Variable(
        initial_value=2, name='v2', collections=[tf.GraphKeys.MODEL_VARIABLES])
    tf.Variable(
        initial_value=np.array([[0.4, 0.3], [0.55, 0.37]]),
        name='v3',
        collections=[tf.GraphKeys.MODEL_VARIABLES])
    variable_name_to_shape = {
        'test/v1': tensor1.get_shape(),
        'v2': tensor2.get_shape(),
        'v3': (2, 3)  # Assigns a wrong shape to tensor 'v3'.
    }
    variables = set(
        image_utils.get_variables_to_restore_from_pretrain_checkpoint(
            exclude_scopes='test', variable_shape_map=variable_name_to_shape))
    self.assertTrue('test/v1' not in variables and 'v2' in variables and
                    'v3' not in variables)

  def test_loss_with_regularization(self):
    weights = np.array([[0.4, 0.3], [0.55, 0.37]])
    bias = np.array([0.2, 0.3])
    groundtruth_labels = np.array([1, 0])
    # Creates two features of dimension two.
    features = np.array([[0.3, 0.3], [0.4, 0.4]])

    expected_total_loss = tf_testing_utils.softmax_cross_entropy_loss(
        features.dot(weights) + bias,
        groundtruth_labels) + 0.1 * np.sum(weights**2) / 2

    feature_vector = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    logits = tf.layers.dense(
        feature_vector,
        units=2,
        kernel_initializer=tf.constant_initializer(weights, verify_shape=True),
        use_bias=True,
        bias_initializer=tf.constant_initializer(bias, verify_shape=True),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    total_loss = image_utils.build_total_loss(
        logits, tf.one_hot(groundtruth_labels, depth=2), label_smoothing=0.0)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      self.assertTrue(
          np.allclose(
              total_loss.eval(feed_dict={feature_vector.name: features}),
              expected_total_loss))

  def test_build_weight_for_label(self):
    labels = tf.constant([-1, 0, -1], dtype=tf.int32)
    with self.test_session():
      self.assertAllEqual(
          image_utils.build_weight_for_label(labels).eval(), [0, 1, 0])

  def test_build_task_loss(self):
    labels = np.array([-1, 0, -1])
    logits = np.array([[0.4, 0.3], [0.55, 0.37], [0.2, 0.8]])
    loss_tensor = image_utils.build_task_loss(
        labels=labels,
        logits=logits,
        task_name='task1',
        task_classes=2,
        label_smoothing=0)
    # Example 0 and 2 are don't-care examples so we compute the loss with
    # example 1.
    with self.test_session():
      self.assertAlmostEqual(
          loss_tensor.eval(),
          tf_testing_utils.softmax_cross_entropy_loss(
              np.array([[0.55, 0.37]]), np.array([0])),
          places=5)

  def test_build_task_weight_from_label(self):
    task_name_to_labels = {'task1': tf.constant(np.array([-1, -1, -1, 1]), dtype=tf.int32),
                           'task2': tf.constant(np.array([ 1,  1, 0, -1]), dtype=tf.int32)}
    task_name_to_weights = image_utils.build_task_weight_from_label(task_name_to_labels)
    with self.test_session():
      self.assertAlmostEqual(task_name_to_weights['task1'].eval(), 0.25)
      self.assertAlmostEqual(task_name_to_weights['task2'].eval(), 0.75)

  def test_build_weighted_task_reg_loss(self):
    task_name_to_labels = {'task1': tf.constant(np.array([-1, -1, -1, 1]), dtype=tf.int32),
                           'task2': tf.constant(np.array([ 1,  1, 0, -1]), dtype=tf.int32)}
    task_name_to_reg_losses = {'task1': tf.constant(50.0, dtype=tf.float32),
                               'task2': tf.constant(30.0, dtype=tf.float32)}
    with self.test_session():
      self.assertAlmostEqual(image_utils.build_weighted_task_reg_loss(task_name_to_reg_losses, task_name_to_labels).eval(), 0.25*50.0+0.75*30.0)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
