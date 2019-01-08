import numpy as np

def _softmax(logits):
  """Computes softmax values for logits.

  Args:
      logits: A num_examples by num_classes array that contains the
        (unnormalized) probability of each example belonging to each class.

  Returns:
      The softmax of input logits.
  """
  exponents = np.exp(logits)
  sum_of_exponents = np.sum(exponents, axis=1)
  element_wise_sum = np.tile(sum_of_exponents,
                             (exponents.shape[1], 1)).transpose()
  return exponents / element_wise_sum


def softmax_cross_entropy_loss(logits, labels):
  """Computes softmax loss given logits and example labels.

  Args:
      logits: A num_examples by num_classes array that contains the
        (unnormalized) probability of each example belonging to each class.
      labels: An array contains the label for each example.

  Returns:
      The softmax cross entropy loss.
  """
  softmax_vals = _softmax(logits)
  log_likelihood = -np.log(softmax_vals[range(labels.shape[0]), labels])
  loss = np.sum(log_likelihood) / labels.shape[0]
  return loss
