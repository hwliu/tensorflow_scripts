import os
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from model_helpers import add_final_layer
from model_helpers import get_probabilities_and_labels_from_logits
from model_helpers import metric_fn
from model_helpers import softmax_cross_entropy_loss
from model_helpers import get_total_loss

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  exps = np.exp(x)  # (n_example, n_classes)
  sums = np.sum(exps, axis=1)  # (n_example, )
  n_classes = exps.shape[1]
  t = np.tile(sums, (n_classes, 1)).transpose()
  return exps / t


def stable_softmax(X):
  exps = np.exp(X - np.max(X))
  return exps / np.sum(exps)


def softmax_cross_entropy(y, X):
  """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of
        labels if required.
    """
  m = y.shape[0]
  p = softmax(X)
  # We use multidimensional array indexing to extract
  # softmax probability of the correct label for each sample.
  # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
  log_likelihood = -np.log(p[range(m), y])
  loss = np.sum(log_likelihood) / m
  return loss


class UtilTest(tf.test.TestCase):
  def setUp(self):
    self._feature_dim = 10
    self._num_classes = 2
    self._weights = np.random.uniform(
        low=-5.0, high=5.0, size=(self._feature_dim, self._num_classes)).astype(float)
    self._bias = np.random.uniform(
        low=-5.0, high=5.0, size=(self._num_classes,)).astype(float)


  def test_add_final_layer(self):
    features = np.random.uniform(
        low=-5.0, high=5.0, size=(1, self._feature_dim)).astype(float)
    feature_vector = tf.constant(features, dtype=tf.float32)
    logits = add_final_layer(
        feature_vector,
        self._num_classes,
        kernel_initializer=tf.constant_initializer(self._weights, verify_shape=True),
        use_bias=True,
        bias_initializer=tf.constant_initializer(self._bias, verify_shape=True),
        activation=tf.nn.softmax)
    init = tf.global_variables_initializer()
    # Get the model weights and bias.
    model_weights = tf.get_default_graph().get_tensor_by_name(
        os.path.split(logits.name)[0] + '/kernel:0')
    model_bias = tf.get_default_graph().get_tensor_by_name(
        os.path.split(logits.name)[0] + '/bias:0')
    # Compute expected result.
    expected_logits = softmax(features.dot(self._weights) + self._bias)
    with self.test_session() as session:
      session.run(init)
      self.assertTrue(np.allclose(model_weights.eval(), self._weights))
      self.assertTrue(np.allclose(model_bias.eval(), self._bias))
      self.assertTrue(np.allclose(logits.eval(), expected_logits))

  def test_cross_entropy_loss_function(self):
    labels = np.array([1, 0, 1, 0, 1, 1])
    logits = np.array([[0.4, 0.3], [0.55, 0.37], [0.2, 0.8], [0.95, 0.9],
                       [0.7, 0.8], [0.3, 0.5]])
    expected_loss = softmax_cross_entropy(labels, logits)

    loss = softmax_cross_entropy_loss(labels, logits)
    init = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init)
      self.assertAlmostEqual(loss.eval(), expected_loss, places=5)

  def test_get_probabilities_and_labels_from_logits(self):
    logits = np.array([[0.4, 0.3], [0.55, 0.37], [0.2, 0.8], [0.95, 0.9],
                       [0.7, 0.8], [0.3, 0.5]])
    predicted_labels, probs = get_probabilities_and_labels_from_logits(logits)
    expected_labels = [0, 0, 1, 0, 1, 1]
    expected_probs = softmax(logits)
    init = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init)
      self.assertTrue(np.allclose(predicted_labels.eval(), expected_labels))
      self.assertTrue(np.allclose(probs.eval(), expected_probs))

  def test_metrics_fn(self):
    ground_truth_labels = np.array([1, 0, 0, 1, 0, 1])
    predicted_labels = np.array([0, 1, 0, 1, 1, 0])
    predicted_probs = np.array([0.3, 0.9, 0.2, 0.7, 0.6, 0.2])
    prs = precision_recall_curve(ground_truth_labels, predicted_probs)
    expected_auc_under_pr = auc(prs[1], prs[0])
    expected_auc_under_roc = roc_auc_score(ground_truth_labels, predicted_probs)
    result = metric_fn(
        tf.constant(ground_truth_labels), tf.constant(predicted_labels),
        predicted_probs)
    init_global_variables = tf.global_variables_initializer()
    init_local_variables = tf.local_variables_initializer()
    with self.test_session() as session:
      session.run(init_global_variables)
      session.run(init_local_variables)
      self.assertAlmostEqual(
          result['auc'][1].eval(), expected_auc_under_roc, places=5)
      self.assertAlmostEqual(
          result['auc_pr'][1].eval(), expected_auc_under_pr, places=5)
      self.assertAlmostEqual(result['accuracy'][1].eval(), 0.33333, places=5)

  def test_regularization_loss(self):
    # Generate 5 examples.
    groundtruth_labels = np.array([1, 0, 1, 1, 0])
    features = np.random.uniform(
        low=-5.0, high=5.0, size=(5, self._feature_dim)).astype(float)
    expected_logits = features.dot(self._weights) + self._bias
    expected_primary_loss = softmax_cross_entropy(groundtruth_labels, expected_logits)
    expected_l2_loss = 0.1* np.sum(self._weights**2) / 2
    expected_total_loss = expected_primary_loss + expected_l2_loss

    feature_vector = tf.placeholder(dtype=tf.float32, shape=[None, self._feature_dim])
    logits = tf.layers.dense(feature_vector,
                             units=self._num_classes,
                             kernel_initializer=tf.constant_initializer(self._weights, verify_shape=True),
                             use_bias=True,
                             bias_initializer=tf.constant_initializer(self._bias, verify_shape=True),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    primary_loss = get_total_loss(tf.losses.sparse_softmax_cross_entropy,
                          groundtruth_labels,
                          logits, enable_regularization=False)
    total_loss = get_total_loss(tf.losses.sparse_softmax_cross_entropy,
                          groundtruth_labels,
                          logits, enable_regularization=True)
    init_global_variables = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init_global_variables)
      self.assertTrue(np.allclose(logits.eval(feed_dict={feature_vector.name: features}), expected_logits))
      self.assertTrue(np.allclose(primary_loss.eval(feed_dict={feature_vector.name: features}), expected_primary_loss))
      self.assertTrue(np.allclose(total_loss.eval(feed_dict={feature_vector.name: features}), expected_total_loss))


if __name__ == '__main__':
  tf.test.main()
