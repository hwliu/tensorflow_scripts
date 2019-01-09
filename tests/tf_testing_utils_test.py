"""Unit tests for utility functions in tf_testing_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

import tf_testing_utils


class TestingUtilityTest(unittest.TestCase):

  def test_softmax_cross_entropy_loss(self):
    # Uses the natural log to undo the effect of exp in the softmax function.
    class_0_probability = np.log(2.5)
    class_1_probability = np.log(5.0)
    # After softmax normalization, we expect the normalized class 0 probability
    # to become 2.5 / (2.5 + 5.0).
    loss = tf_testing_utils.softmax_cross_entropy_loss(
        np.array([[class_0_probability, class_1_probability]]), np.array([0]))
    # Given the label being 0, only the normalized class 0 probability is used
    # to compute the negative log sum.
    self.assertAlmostEqual(loss, -np.log(2.5 / (2.5 + 5.0)))

    # Likewise when the label is 1.
    loss = tf_testing_utils.softmax_cross_entropy_loss(
        np.array([[class_0_probability, class_1_probability]]), np.array([1]))
    self.assertAlmostEqual(loss, -np.log(5.0 / (2.5 + 5.0)))

  def test_softmax_cross_entropy_loss_with_dont_care(self):
    # After softmax normalization, we expect the normalized class 0 probability
    # to become 2.5 / (2.5 + 5.0).
    loss = tf_testing_utils.softmax_cross_entropy_loss(
        np.array([[np.log(2.5), np.log(5.0)], [np.log(5.0),
                                               np.log(5.0)]]), np.array([0,
                                                                         -1]))
    # Given the label being 0, only the normalized class 0 probability is used
    # to compute the negative log sum while second element is ignored.
    self.assertAlmostEqual(loss, -np.log(2.5 / (2.5 + 5.0)))

    # All elements are to be ignored.
    loss = tf_testing_utils.softmax_cross_entropy_loss(
        np.array([[np.log(2.5), np.log(5.0)], [np.log(5.0),
                                               np.log(5.0)]]),
        np.array([-1, -1]))
    self.assertAlmostEqual(loss, 0)


if __name__ == '__main__':
  unittest.main()
