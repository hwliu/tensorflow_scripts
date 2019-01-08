"""Multi-head utils tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import multi_task_utils
import tf_testing_utils
import visual_shopping_image_utils as image_utils

FEATURE_KEY = 'feature/floats'

def _create_input_fn(features, taskname_and_labels):

  def _input_fn(params):
    del params
    tensor_tuples =(features,)
    task_names = []
    for taskname_and_label in taskname_and_labels:
       tensor_tuples = tensor_tuples + (taskname_and_label[1],)
       task_names.append(taskname_and_label[0])
    dataset = tf.data.Dataset.from_tensor_slices(tensor_tuples)
    raw_features = dataset.batch(4).repeat(1).make_one_shot_iterator().get_next()
    features_batch = features_batch = {FEATURE_KEY: raw_features[0]}

    labels_batch = {}
    for task_name, label in zip(task_names, raw_features[1:]):
        labels_batch[task_name] = label
    return features_batch, labels_batch

  return _input_fn


class AssertLossHook(tf.train.SessionRunHook):

  def __init__(self, unittest_object, expected_loss):
    self._unittest_object = unittest_object
    self._expected_loss = expected_loss

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(
        run_context.session.graph.get_tensor_by_name('total_loss:0'))

  def after_run(self, run_context, run_values):
    self._unittest_object.assertAlmostEqual(run_values.results,
                                            self._expected_loss)
    return 0


class MultiTaskUtilTest(tf.test.TestCase):

  def test_input_fn(self):
    ## TODO(b/xxx) : shuffle the files.
    input_fn = _create_input_fn(features=np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8], [0.9, 0.7]]),
                                taskname_and_labels = [('task1', np.array([-1, -1, -1, -1])), ('task2', np.array([1,0,0,1]))])
    params = []
    _, labels = input_fn(params)

    with self.cached_session() as sess:
      labels = sess.run(labels)
      self.assertAllEqual([-1, -1, -1, -1], labels['task1'])
      self.assertAllEqual([1, 0, 0, 1], labels['task2'])

  def test_training_multi_task(self):
    features = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8], [0.9, 0.7]])
    input_fn = _create_input_fn(features=features,
                                taskname_and_labels = [('task1', np.array([-1, -1, -1, -1])), ('task2', np.array([1,0,0,1]))])
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.create_estimator_spec_fn(tasknames_to_num_classes, kernels))
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    expected_loss = tf_testing_utils.softmax_cross_entropy_loss(
                    features.dot(kernels['task2']),
                    np.array([1, 0, 0, 1])) + 0.1 * np.sum(kernels['task1']**2) / 2 + 0.1 * np.sum(kernels['task2']**2) / 2

    estimator.train(
        input_fn=input_fn,
        steps=1,
        hooks=[
            AssertLossHook(
                self, expected_loss
                )
        ])

    self.assertAllClose(
        estimator.get_variable_value('task1_logit/kernel'),
        np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]))
    self.assertAllClose(
        estimator.get_variable_value('task2_logit/kernel'),
        np.array([[0.2, 0.4], [0.3, 0.5]]))

def test_evaluating_multi_task(self):
    features = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8], [0.9, 0.7]])
    input_fn = _create_input_fn(features=features,
                                taskname_and_labels = [('task1', np.array([-1, -1, -1, -1])), ('task2', np.array([1,0,0,1]))])
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.create_estimator_spec_fn(tasknames_to_num_classes, kernels))
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    expected_loss = tf_testing_utils.softmax_cross_entropy_loss(
                    features.dot(kernels['task2']),
                    np.array([1, 0, 0, 1])) + 0.1 * np.sum(kernels['task1']**2) / 2 + 0.1 * np.sum(kernels['task2']**2) / 2

    result_metrics = estimator.evaluate(
        input_fn=input_fn,
        steps=1,
        hooks=[
            AssertLossHook(
                self, expected_loss
                )
        ])

    print(result_metrics)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
