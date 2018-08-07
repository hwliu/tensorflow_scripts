def get_model_fn_tpu(num_categories=10,
                 dropout_rate=0.2,
                 learning_rate=0.001):
  depth_mul = 1.0
  optimizer_to_use = 'sgd'

  def inception_v3_tpu_model_fn(features, labels, mode, params):
    """Model function for inception V3."""
    is_training_mode = (mode == tf.estimator.ModeKeys.TRAIN)
    is_predict_mode = (mode == tf.estimator.ModeKeys.PREDICT)
    image = adjust_image(features['x'])
    logits, endpoints = inception_v3(
          image,
          num_classes=num_categories,
          dropout_keep_prob=1.0 - dropout_rate,
          is_training=is_training_mode,
          depth_multiplier=depth_mul)
    tf.logging.info('Num inception layers = %d', len(endpoints))

    predictions = endpoints
    predictions.update({
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    })
    if is_predict_mode:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    host_call = None
    train_op = None
    if is_training_mode:
      optimizer = get_optimizer(optimizer_to_use, learning_rate)
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, host_call=host_call)

    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])
    }
    eval_metrics = (metric_fn, [labels, logits])
    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, host_call=host_call, )



    return 0

  return 0

def create_inception_v3_models(use_tpu=False,
                               num_categories=10,
                               dropout_rate=0.2,
                               learning_rate=0.001):
    if not use_tpu:
      return get_model_fn(num_categories, dropout_rate, learning_rate)
    else:
      return None

    return 0
