from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join
from absl import flags
import tensorflow as tf
import model_utils
import dataset

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

flags.DEFINE_integer('total_training_steps', 50, 'Number of training steps.')

flags.DEFINE_integer('export_model_steps', 5,
                     'Number of training steps to export models')

flags.DEFINE_integer('batch_size', 100, 'Batch size for training/eval.')

flags.DEFINE_string('dataset_config_json',
                    '/media/haoweiliu/Data/tensorflow_scripts/dataset/watermark.json',
                    'Json file that configures the datasets.')
flags.DEFINE_string('model_name',
                    'inception_v3', 'name of the model/network.')

def run_training_eval_pipeline(model_name,
                               dataset_config_json,
                               output_model_dir):
  input_dataset = dataset.create_dataset_from_json_file(dataset_config_json)
  train_input_fn = 
  hparams = tf.contrib.training.HParams(learning_rate=0.01)
  run_config = tf.estimator.RunConfig(save_summary_steps=100)
  training_preprocessing_fn = model_utils.select_preprocessing_fn(model_name,
                                                         is_training=True)
  eval_preprocessing_fn = model_utils.select_preprocessing_fn(model_name,
                                                         is_training=False)
  network_fn = model_utils.select_network(model_name,
                               input_dataset.num_classes, is_training=True)
  model_fn = model_utils.create_model_fn(network_fn,
                                         training_preprocessing_fn,
                                         eval_preprocessing_fn,
                                         input_dataset.num_classes,
                                         input_dataset.is_multilabel,
                                         num_of_replicas=1,
                                         learning_rate=hparams.learning_rate)

  ## add warm_start
  classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=output_model_dir,
      config=run_config)

  ## add exporters
  #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
  #                                    max_steps=300000)
  #eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
  #tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


def main(unused_argv):
  run_training_eval_pipeline(FLAGS.model_name,
                             FLAGS.dataset_config_json,
                             FLAGS.output_model_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

