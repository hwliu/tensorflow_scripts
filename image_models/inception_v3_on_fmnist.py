from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model_helpers import get_model_fn
#from google3.experimental.users.haoweiliu.watermark_training.model_helpers import get_model_fn
from PIL import Image
import PIL

FLAGS = flags.FLAGS

flags.DEFINE_string('output_model_dir', '/tmp/inception_on_fmnist',
                    'The directory to save the trained model.')

flags.DEFINE_float('dropout_rate', 0.2,
                   'Dropout rate to use when training the network.')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate to use when training the network.')

flags.DEFINE_integer('total_training_steps', 500,
                     'Number of training steps.')

flags.DEFINE_integer('export_model_steps', 100,
                     'Number of training steps to export models')

flags.DEFINE_integer('batch_size', 100,
                     'Batch size for training/eval.')

def adjust_image(data):
  print(data)

  data = tf.to_float(data)
  data = tf.subtract(data, 128.0)
  data = tf.divide(data, 255.0)
  # Reshape to [batch, height, width, channels].
  imgs = tf.reshape(data, [-1, 28, 28, 1])
  print(imgs)

  # Adjust image size to Inception-v3 input.
  imgs = tf.image.resize_images(imgs, (299, 299))
  print(imgs)
  exit()
  # Convert to RGB image.
  imgs = tf.image.grayscale_to_rgb(imgs)
  return imgs

def main(unused_argv):
  # Load training and eval data.
  fashion_mnist = keras.datasets.fashion_mnist
  (train_data, train_labels), (eval_data,
                               eval_labels) = fashion_mnist.load_data()
  train_labels = train_labels.astype(np.int32)
  eval_labels = eval_labels.astype(np.int32)

  # Set up input function.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=train_data,
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True)

  # Create the Estimator.
  run_config = tf.estimator.RunConfig(save_summary_steps=10)
  inception_model_fn = get_model_fn(10, adjust_image, FLAGS.dropout_rate, FLAGS.learning_rate)
  inception_classifier = tf.estimator.Estimator(
      model_fn=inception_model_fn, model_dir=FLAGS.output_model_dir, config=run_config)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {'probabilities': 'softmax_tensor'}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  for cycle in range(FLAGS.total_training_steps // FLAGS.export_model_steps):
    tf.logging.info('Starting training cycle %d.' % cycle)
    inception_classifier.train(
      input_fn=train_input_fn, steps=FLAGS.export_model_steps, hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = inception_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
