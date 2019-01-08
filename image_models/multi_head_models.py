import tensorflow as tf

def get_model_fn():
  def model_fn(images, labels, mode):
    # Make a shared layer.
    features = tf.layers.dense(images, units = 1)

    # Make two logit heads.
    logit1 = tf.layers.dense(features, units = 2)
    logit2 = tf.layers.dense(features, units = 3)

    # Create simple heads and specify head name.
    head1 = multi_class_head(n_classes=3, name='head1')
    head2 = binary_classification_head(name='head2')
    # Create multi-head from two simple heads.
    head = multi_head([head1, head2])

    return head.create_estimator_spec(features, mode, logits, labels)


  return model_fn
