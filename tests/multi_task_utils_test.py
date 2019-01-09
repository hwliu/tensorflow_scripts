"""Multi-task utils unit tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

import multi_task_utils
import tf_testing_utils


def _create_input_fn(features, taskname_and_labels):

  def _input_fn(params):
    del params
    # The export function of Estimator only allows float32 as the intput type
    # for floating point for the exported model. Here we specifically make a
    # tensor of float type for our test data so tensorflow won't up-cast all
    # model parameters to double during training. Normally, we would have to
    # specify the feature spec when implementing the input_fn to parse the
    # input SSTable.
    tensor_tuples = (tf.constant(features, dtype=tf.float32),)
    task_names = []
    for taskname_and_label in taskname_and_labels:
      tensor_tuples = tensor_tuples + (taskname_and_label[1],)
      task_names.append(taskname_and_label[0])
    dataset = tf.data.Dataset.from_tensor_slices(tensor_tuples)
    # Uses batch size 4 and repeats only once.
    dataset = dataset.batch(4).repeat(1)
    raw_features = dataset.make_one_shot_iterator().get_next()

    labels_batch = {}
    for task_name, label in zip(task_names, raw_features[1:]):
      labels_batch[task_name] = label
    return {multi_task_utils.FEATURE_KEY: raw_features[0]}, labels_batch

  return _input_fn


# The train function of Estimator does not return any internal states, including
# the training loss. The only way to get the training loss is implementing a
# hook, that is called by tensorflow at every training step. We implement the
# hook, that takes a unit test object and a dictionary of tensor name to their
# expected value and makes assertion at every training step.
class AssertLossHook(tf.train.SessionRunHook):

  def __init__(self, unittest, tensor_name_to_expected_value):
    self._unittest = unittest
    self._tensor_name_to_expected_value = tensor_name_to_expected_value

  def before_run(self, run_context):
    tensors_to_fetch = []
    for tensor_name in self._tensor_name_to_expected_value.keys():
      tensors_to_fetch.append(run_context.session.graph.get_tensor_by_name(tensor_name))
    return tf.train.SessionRunArgs(tensors_to_fetch)

  def after_run(self, run_context, run_values):
    for result, expected_value in zip(run_values.results, self._tensor_name_to_expected_value.values()):
       self._unittest.assertAlmostEqual(result, expected_value)
    return 0


class MultiTaskUtilTest(tf.test.TestCase):

  def test_input_fn(self):
    input_fn = _create_input_fn(
        features=np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8],
                           [0.9, 0.7]]),
        taskname_and_labels=[('task1', np.array([-1, -1, -1, -1])),
                             ('task2', np.array([1, 0, 0, 1]))])
    params = []
    _, labels = input_fn(params)

    with self.cached_session() as sess:
      labels = sess.run(labels)
      self.assertAllEqual([-1, -1, -1, -1], labels['task1'])
      self.assertAllEqual([1, 0, 0, 1], labels['task2'])

  def test_multi_task_training_total_loss(self):
    features = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8], [0.9, 0.7]])
    input_fn = _create_input_fn(
        features=features,
        taskname_and_labels=[('task1', np.array([-1, -1, -1, -1])),
                             ('task2', np.array([1, 0, 0, 1]))])
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    tasknames_to_kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.multi_task_estimator_spec_fn, tasknames_to_kernels,
        tasknames_to_num_classes, no_optimizer=True)
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    # Since there is no valid examples for task1, the total loss is equal to
    # the cross_entropy loss plus the regularization loss for task1.
    expected_loss = tf_testing_utils.softmax_cross_entropy_loss(
        features.dot(tasknames_to_kernels['task2']), np.array([
            1, 0, 0, 1
        ])) + 0.1 * np.sum(tasknames_to_kernels['task2']**2) / 2

    tensor_name_to_expected_value = {'total_loss:0':expected_loss}
    estimator.train(
        input_fn=input_fn, steps=1, hooks=[AssertLossHook(self, tensor_name_to_expected_value)])

    self.assertAllClose(
        estimator.get_variable_value('task1_logit/kernel'),
        np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]))
    self.assertAllClose(
        estimator.get_variable_value('task2_logit/kernel'),
        np.array([[0.2, 0.4], [0.3, 0.5]]))

  def test_multi_task_training_with_regularization_loss(self):
    features = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8], [0.9, 0.7]])
    input_fn = _create_input_fn(
        features=features,
        taskname_and_labels=[('task1', np.array([-1, -1, -1, -1])),
                             ('task2', np.array([-1, -1, -1, -1]))])
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    tasknames_to_kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.multi_task_estimator_spec_fn, tasknames_to_kernels,
        tasknames_to_num_classes, no_optimizer=True)
    estimator = tf.estimator.Estimator(model_fn=model_fn)

    tensor_name_to_expected_value = {'task1_reg_loss:0' : 0.1 * np.sum(tasknames_to_kernels['task1']**2) / 2,
                                     'task2_reg_loss:0' : 0.1 * np.sum(tasknames_to_kernels['task2']**2) / 2}

    estimator.train(
        input_fn=input_fn, steps=1, hooks=[AssertLossHook(self, tensor_name_to_expected_value)])
    # Given that we do not provide any training examples for task1, kernel1
    # should stay at the initial values.
    self.assertAllClose(
        estimator.get_variable_value('task1_logit/kernel'),
        np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]))

  def test_multi_task_training_with_optimization(self):
    features = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8], [0.9, 0.7]])
    input_fn = _create_input_fn(
        features=features,
        taskname_and_labels=[('task1', np.array([-1, -1, -1, -1])),
                             ('task2', np.array([-1, -1, -1, -1]))])
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    tasknames_to_kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.multi_task_estimator_spec_fn, tasknames_to_kernels,
        tasknames_to_num_classes)
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    expected_loss = 0.1 * np.sum(tasknames_to_kernels['task1']**2) / 2 + 0.1 * np.sum(
            tasknames_to_kernels['task2']**2) / 2
    estimator.train(
        input_fn=input_fn, steps=1)

    # Given that we do not provide any training examples for task1, kernel1
    # should stay at the initial values.
    self.assertAllClose(
        estimator.get_variable_value('task1_logit/kernel'),
        np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]))

  def test_multi_task_evaluation(self):
    features = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8], [0.9, 0.95]])
    taskname_and_labels = [('task1', np.array([0, 2, -1, -1])),
                           ('task2', np.array([-1, -1, 0, 1]))]
    input_fn = _create_input_fn(
        features=features, taskname_and_labels=taskname_and_labels)
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    tasknames_to_kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.multi_task_estimator_spec_fn, tasknames_to_kernels,
        tasknames_to_num_classes)
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    expected_loss = tf_testing_utils.softmax_cross_entropy_loss(
        features.dot(tasknames_to_kernels['task1']), taskname_and_labels[0]
        [1]) + tf_testing_utils.softmax_cross_entropy_loss(
            features.dot(tasknames_to_kernels['task2']),
            taskname_and_labels[1][1]) + 0.5 * 0.1 * np.sum(
                tasknames_to_kernels['task1']**2) / 2 + 0.5 * 0.1 * np.sum(
                    tasknames_to_kernels['task2']**2) / 2

    result_metrics = estimator.evaluate(input_fn=input_fn, steps=1)
    self.assertAlmostEqual(result_metrics['loss'], expected_loss, places=6)
    self.assertAlmostEqual(result_metrics['task1/Eval/Accuracy/validation'],
                           0.5)
    self.assertAlmostEqual(result_metrics['task2/Eval/Accuracy/validation'],
                           0.5)

  def test_multi_task_prediction(self):
    features = np.array([[0.2, 0.4]])
    taskname_and_labels = [('task1', np.array([0])), ('task2', np.array([1]))]
    input_fn = _create_input_fn(
        features=features, taskname_and_labels=taskname_and_labels)
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    tasknames_to_kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.multi_task_estimator_spec_fn, tasknames_to_kernels,
        tasknames_to_num_classes)
    estimator = tf.estimator.Estimator(model_fn=model_fn)

    # Estimator.predict returns an iterator for every input example. Given that
    # we have passed one example, prediction_result should not be None.
    prediction_result = next(estimator.predict(input_fn=input_fn), None)

    with self.cached_session():
      self.assertAllClose(
          prediction_result['task1/probabilities'],
          tf.squeeze(
              tf.nn.softmax(features.dot(
                  tasknames_to_kernels['task1']))).eval())
      self.assertAllClose(
          prediction_result['task2/probabilities'],
          tf.squeeze(
              tf.nn.softmax(features.dot(
                  tasknames_to_kernels['task2']))).eval())
      self.assertEqual(prediction_result['task1/top_class'], 2)
      self.assertEqual(prediction_result['task2/top_class'], 1)

  def test_multi_task_model_export(self):
    model_dir = '/media/haoweiliu/Data/tensorflow_scripts/tests/models'
    export_dir = model_dir + '/exports'
    checkpoint_path = model_dir + '/model.ckpt-0'
    features = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.8],
                         [0.9, 0.7]]).astype(float)
    input_fn = _create_input_fn(
        features=features,
        taskname_and_labels=[('task1', np.array([-1, -1, -1, -1])),
                             ('task2', np.array([1, 0, 0, 1]))])
    tasknames_to_num_classes = {'task1': 3, 'task2': 2}
    tasknames_to_kernels = {
        'task1': np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]]),
        'task2': np.array([[0.2, 0.4], [0.3, 0.5]])
    }
    model_fn = multi_task_utils.create_model_fn(
        multi_task_utils.multi_task_estimator_spec_fn, tasknames_to_kernels,
        tasknames_to_num_classes)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)
    estimator.train(input_fn=input_fn, steps=1)

    def serving_input_fn():
      input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
          {multi_task_utils.FEATURE_KEY: tf.FixedLenFeature([2], tf.float32)},
          default_batch_size=None)
      raw_features, receiver_tensors, _ = input_fn()
      return tf.estimator.export.ServingInputReceiver(raw_features,
                                                      receiver_tensors)

    exported_model_path = estimator.export_saved_model(
        export_dir_base=export_dir,
        checkpoint_path=checkpoint_path,
        serving_input_receiver_fn=serving_input_fn)
    self.assertTrue(os.path.isfile(exported_model_path + '/saved_model.pb'))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
