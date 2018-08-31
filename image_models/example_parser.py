"""Utility functions to process the input for inception v3 model."""
import tensorflow as tf


class Parser(object):
  def __init__(self, key_to_features):
    self._key_to_features_map = key_to_features

  def return_images_labels(self, parsed_features):
    ## todo: cleanup
    feature0 = parsed_features[self._key_to_features_map.keys()[0]]
    feature1 = parsed_features[self._key_to_features_map.keys()[1]]
    images = feature0 if feature0.dtype == tf.string else feature1
    labels = feature0 if feature0.dtype == tf.int64 else feature1
    return images, labels

class TFSequenceExampleParser(Parser):

  def parse_single_example(self, value):
    parsed_features, _ = tf.parse_single_sequence_example(
      value, context_features=self._key_to_features_map)
    return super(TFSequenceExampleParser, self).return_images_labels(parsed_features)

class TFExampleParser(Parser):

  def parse_single_example(self, value):
    parsed_features = tf.parse_single_example(
      value, features=self._key_to_features_map)
    return super(TFExampleParser, self).return_images_labels(parsed_features)

def create_dataset_parser(parser_type, key_to_features):
  if parser_type == 'TFSequenceExample':
    return TFSequenceExampleParser(key_to_features)
  elif parser_type == 'TFExample':
    return TFSequenceExampleParser(key_to_features)
  else:
    return None

