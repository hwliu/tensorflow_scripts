"""Code for training inception V3 model on watermark dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
import tensorflow as tf
#from google3.experimental.users.haoweiliu.watermark_training.input_processing_helpers import create_export_model_signature
#from google3.experimental.users.haoweiliu.watermark_training.input_processing_helpers import create_input_fn_for_images_sstable
#from google3.experimental.users.haoweiliu.watermark_training.model_helpers import get_model_fn
from model_helpers import get_model_fn
from input_preprocessing_helpers import create_input_fn_for_images_sstable
from input_preprocessing_helpers import create_export_model_signature

FLAGS = flags.FLAGS

flags.DEFINE_string('output_model_dir', '/tmp/inception_on_fmnist',
                    'The directory to save the trained model.')

flags.DEFINE_string(
    'training_dataset_path',
    '/cns/oy-d/home/visual-shopping-training/haoweiliu/watermark_project/'
    'way_data/training_dataset@500',
    'The path to the sstable that holds the training data.')

flags.DEFINE_string(
    'validation_dataset_path',
    '/cns/oy-d/home/visual-shopping-training/haoweiliu/watermark_project/'
    'way_data/validation_dataset@500',
    'The path to the sstable that holds the validation data.')

flags.DEFINE_string(
    'testing_dataset_path',
    '/cns/oy-d/home/visual-shopping-training/haoweiliu/watermark_project/'
    'way_data/testing_dataset@500',
    'The path to the sstable that holds the test data.')

flags.DEFINE_float('dropout_rate', 0.2,
                   'Dropout rate to use when training the network.')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate to use when training the network.')

flags.DEFINE_integer('total_training_steps', 3, 'Number of training steps.')

flags.DEFINE_integer('export_model_steps', 1,
                     'Number of training steps to export models')

flags.DEFINE_integer('batch_size', 100, 'Batch size for training/eval.')


def main(unused_argv):

  # Create the Estimator.
  run_config = tf.estimator.RunConfig(save_summary_steps=10)
  inception_model_fn = get_model_fn(10, None, FLAGS.dropout_rate,
                                    FLAGS.learning_rate)
  inception_classifier = tf.estimator.Estimator(
      model_fn=inception_model_fn,
      model_dir=FLAGS.output_model_dir,
      config=run_config)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {'probabilities': 'softmax_tensor'}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  for cycle in range(FLAGS.total_training_steps // FLAGS.export_model_steps):
    tf.logging.info('Starting training cycle %d.' % cycle)
    inception_classifier.train(
        input_fn=create_input_fn_for_images_sstable(
            FLAGS.training_dataset_path, mode=tf.estimator.ModeKeys.TRAIN),
        steps=FLAGS.export_model_steps,
        hooks=[logging_hook])

    #inception_classifier.export_savedmodel(FLAGS.output_model_dir + str(cycle),
    #                                       create_export_model_signature())

    eval_results = inception_classifier.evaluate(
        input_fn=create_input_fn_for_images_sstable(
            FLAGS.validation_dataset_path, mode=tf.estimator.ModeKeys.EVAL))
    print(eval_results)

  test_results = inception_classifier.evaluate(
      input_fn=create_input_fn_for_images_sstable(
          FLAGS.testing_dataset_path, mode=tf.estimator.ModeKeys.EVAL))
  print(test_results)

  return 0


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
