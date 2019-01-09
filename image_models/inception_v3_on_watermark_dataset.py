"""Code for training inception V3 model on watermark dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join
from absl import flags
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.contrib import predictor
from tensorflow.python import pywrap_tensorflow
import tensorflow_hub as hub
#from google3.experimental.users.haoweiliu.watermark_training.inception_v3_on_watermark_settings import INPUT_FEATURE_NAME
#from google3.experimental.users.haoweiliu.watermark_training.input_processing_helpers import create_input_fn_for_images_sstable
#from google3.experimental.users.haoweiliu.watermark_training.input_processing_helpers import exported_model_input_signature
#from google3.experimental.users.haoweiliu.watermark_training.model_helpers import get_model_fn
from inception_v3_on_watermark_settings import INPUT_FEATURE_NAME
from inception_v3_on_watermark_settings import KEYS_TO_FEATURE_MAP
from inception_v3_on_watermark_settings import CERVALT_KEYS_TO_FEATURE_MAP

from input_preprocessing_helpers import create_input_fn_for_images_sstable
from input_preprocessing_helpers import exported_model_input_signature
from model_helpers import get_model_fn
from model_helpers import get_raw_model_fn_with_pretrained_model
from tensorflow.python.tools import inspect_checkpoint as chkp

FLAGS = flags.FLAGS

flags.DEFINE_string('output_model_dir', '/media/haoweiliu/Data/scratch_models/inception_v3_on_watermark',
                    'The directory to save the trained model.')

flags.DEFINE_string(
    'training_dataset_path',
    '/media/haoweiliu/Data/tensorflow_scripts/dataset/cervelat_training_data-00184-of-00207',
    'The path to the sstable that holds the training data.')

flags.DEFINE_string(
    'validation_dataset_path',
    '/media/haoweiliu/Data/tensorflow_scripts/dataset/cervelat_training_data-00184-of-00207',
    'The path to the sstable that holds the validation data.')

flags.DEFINE_string(
    'testing_dataset_path',
    '/media/haoweiliu/Data/tensorflow_scripts/dataset/cervelat_training_data-00184-of-00207',
    'The path to the sstable that holds the test data.')

flags.DEFINE_string(
    'testing_image_path',
    '/media/haoweiliu/Data/tensorflow_scripts/image_models/00001fdb943687e3.jpg',
    'The path to the sstable that holds the test data.')

flags.DEFINE_float('dropout_rate', 0.2,
                   'Dropout rate to use when training the network.')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate to use when training the network.')

flags.DEFINE_integer('total_training_steps', 50, 'Number of training steps.')

flags.DEFINE_integer('export_model_steps', 5,
                     'Number of training steps to export models')

flags.DEFINE_integer('batch_size', 100, 'Batch size for training/eval.')

flags.DEFINE_bool('retrain_inception_model', False,
                  'Whether or not to re-train the network weights.')

flags.DEFINE_string('dataset_type', 'TFExample',
                  'TFExample or TFSequenceExample')

flags.DEFINE_string('optimizer', 'sgd',
                  'sgd, rms or momentum')



def test_savedmodel_with_image(model_dir, test_image_path):
  with open(test_image_path, 'r') as test_image_file:
    data = test_image_file.read()
    tf.logging.info('testing saved model................')
    tf.logging.info(model_dir)
    predict_fn = predictor.from_saved_model(model_dir)
    test_dataitem = []
    test_dataitem.append(data)
    results = predict_fn({INPUT_FEATURE_NAME: test_dataitem})
    print(results)

def get_tensors_and_values_from_checkpoint(file_name):
  tensor_name_to_value = {}
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor_name_to_value[key] = reader.get_tensor(key)
  except Exception as e:
    print(str(e))
  return tensor_name_to_value

def check_variables(key_name, dictoinary):
  if (key_name not in dictoinary):
    print ('no {}'.format(key_name))

def my_inputfunc():
  with open(FLAGS.testing_image_path, 'r') as test_image_file:
    data = test_image_file.read()
  image = {INPUT_FEATURE_NAME: tf.reshape(data, shape=(-1,))}
  label = tf.reshape(1, shape=(-1,))
  return image, label

def main(unused_argv):
   # Create the Estimator.
  run_config = tf.estimator.RunConfig(save_summary_steps=10)

  inception_model_fn = get_model_fn(
      total_steps=1000,
      num_categories=2,
      input_processor=None,
      learning_rate=FLAGS.learning_rate,
      retrain_model=FLAGS.retrain_inception_model,
      optimizer_to_use=FLAGS.optimizer)
  inception_classifier = tf.estimator.Estimator(
      model_fn=inception_model_fn,
      model_dir=FLAGS.output_model_dir,
      config=run_config)

  keys_to_feature_map = CERVALT_KEYS_TO_FEATURE_MAP if FLAGS.dataset_type=='TFExample' else KEYS_TO_FEATURE_MAP
  training_input_fn=create_input_fn_for_images_sstable(
            FLAGS.training_dataset_path,
            keys_to_features_map=keys_to_feature_map,
            dataset_type=FLAGS.dataset_type,
            mode=tf.estimator.ModeKeys.TRAIN,
            batch_size=FLAGS.batch_size)
  validation_input_fn=create_input_fn_for_images_sstable(
            FLAGS.validation_dataset_path,
            keys_to_features_map=keys_to_feature_map,
            dataset_type=FLAGS.dataset_type,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=FLAGS.batch_size)
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {'probabilities': 'softmax_tensor'}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  for cycle in range(FLAGS.total_training_steps // FLAGS.export_model_steps):
    tf.logging.info('Starting training cycle %d.' % cycle)
    inception_classifier.train(
        input_fn=training_input_fn,
        steps=FLAGS.export_model_steps,
        hooks=[logging_hook])

    tf.logging.info('Exporting to saved model: %d.' % cycle)
    saved_model_dir = join(FLAGS.output_model_dir, 'saved_model')
    export_dir = inception_classifier.export_savedmodel(
        saved_model_dir, exported_model_input_signature)
    tf.logging.info('Saved model to %s.' % export_dir)

    eval_results = inception_classifier.evaluate(input_fn=validation_input_fn)
    print(eval_results)

  return 0


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
