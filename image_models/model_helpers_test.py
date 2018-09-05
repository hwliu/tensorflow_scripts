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

  def test_add_final_layer(self):
    feature_dim = 10
    num_classes = 2
    features = np.random.uniform(
        low=-5.0, high=5.0, size=(feature_dim, 1)).astype(float)
    weights = np.random.uniform(
        low=-5.0, high=5.0, size=(1, num_classes)).astype(float)
    bias = np.random.uniform(
        low=-5.0, high=5.0, size=(num_classes,)).astype(float)
    feature_vector = tf.constant(features, dtype=tf.float32)
    logits = add_final_layer(
        feature_vector,
        num_classes,
        kernel_initializer=tf.constant_initializer(weights, verify_shape=True),
        use_bias=True,
        bias_initializer=tf.constant_initializer(bias, verify_shape=True),
        activation=tf.nn.softmax)
    init = tf.global_variables_initializer()
    # Get the model weights and bias.
    model_weights = tf.get_default_graph().get_tensor_by_name(
        os.path.split(logits.name)[0] + '/kernel:0')
    model_bias = tf.get_default_graph().get_tensor_by_name(
        os.path.split(logits.name)[0] + '/bias:0')
    # Compute expected result.
    expected_logits = softmax(features * weights + bias)
    with self.test_session() as session:
      session.run(init)
      self.assertTrue(np.allclose(model_weights.eval(), weights))
      self.assertTrue(np.allclose(model_bias.eval(), bias))
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


if __name__ == '__main__':
  tf.test.main()
