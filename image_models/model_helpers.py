"""Define the model for inception v3 model."""
from input_preprocessing_helpers import preprocess_image
import tensorflow as tf
import tensorflow_hub as hub
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INCEPTION_V3_MODULE_PATH
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INCEPTION_V3_TARGET_IMAGE_SIZE
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INPUT_FEATURE_NAME
from inception_v3_on_watermark_settings import INCEPTION_V3_MODULE_PATH
from inception_v3_on_watermark_settings import INCEPTION_V3_TARGET_IMAGE_SIZE
from inception_v3_on_watermark_settings import INPUT_FEATURE_NAME


def get_optimizer(optimizer_name, learning_rate):
  """Return an optimizer by name."""
  if optimizer_name == 'sgd':
    tf.logging.info('Using SGD optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    tf.logging.info('Using Momentum optimizer')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9)
  elif optimizer_name == 'rms':
    tf.logging.info('Using RMSPropOptimizer')
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  else:
    tf.logging.fatal('Unknown optimizer:', optimizer_name)
    optimizer = None

  return optimizer


def processing_model_input(input_feature):
  image = preprocess_image(
      input_feature[INPUT_FEATURE_NAME],
      tf.constant(INCEPTION_V3_TARGET_IMAGE_SIZE, dtype=tf.int32))
  return image


def get_model_fn(num_categories,
                 input_processor,
                 learning_rate=0.001,
                 retrain_model=False):
  """Wrapper of the inception v3 model function."""
  optimizer_to_use = 'sgd'

  def inception_v3_model_fn(features, labels, mode):
    """Model function for inception V3."""
    is_training_mode = (mode == tf.estimator.ModeKeys.TRAIN)
    is_predict_mode = (mode == tf.estimator.ModeKeys.PREDICT)
    # Load Inception-v3 model.
    inception_v3_module = hub.Module(
        INCEPTION_V3_MODULE_PATH, trainable=retrain_model)
    if input_processor is not None:
      images = input_processor(features)
    else:
      images = tf.map_fn(processing_model_input, features, dtype=tf.float32)

    outputs = inception_v3_module(images)
    logits = tf.layers.dense(inputs=outputs, units=num_categories)
    labels = tf.argmax(input=logits, axis=1)
    probs = tf.nn.softmax(logits, name='softmax_tensor')
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': labels,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'scores': probs
    }

    if is_predict_mode:
      export_outputs = {
          'classify': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    if is_training_mode:
      optimizer = get_optimizer(optimizer_to_use, learning_rate)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def metric_fn(labels, logits):
      predictions = tf.argmax(logits, 1)
      return {
          'accuracy':
              tf.metrics.precision(labels=labels, predictions=predictions),
      }

    metrics_ops = metric_fn(labels, logits)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=metrics_ops)

  return inception_v3_model_fn
