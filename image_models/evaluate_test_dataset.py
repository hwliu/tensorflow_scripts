from absl import flags
import logging
import tensorflow as tf
from tensorflow.contrib import predictor
from input_preprocessing_helpers import shardedfile_to_filelist
from input_preprocessing_helpers import parser
from inception_v3_on_watermark_settings import KEYS_TO_FEATURE_MAP

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/media/haoweiliu/Data/scratch_models/inception_v3_on_watermark/saved_model/1535501631',
                    'The directory containing the trained model.')

flags.DEFINE_string(
    'dataset_path',
    '/media/haoweiliu/Data/tensorflow_scripts/dataset/input_test.tfrecords',
    'The path to the sstable that holds the data for evaluation.')

def run(test_dataset_path, keys_to_features, batch_size, model_dir):
  input_shard_file_list = shardedfile_to_filelist(test_dataset_path)
  parse_fn = (lambda x: parser(x, keys_to_features))
  dataset = tf.data.Dataset(input_shard_file_list)
  dataset = dataset.repeat(1)
  dataset = dataset.map(parse_fn, 32).batch(batch_size)
  predict_fn = predictor.from_saved_model(model_dir)
  while True:
    try:
       images, labels = dataset.make_one_shot_iterator().get_next()
    except tf.errors.OutOfRangeError:
       break
    predictor_fn({FLAGS.input_name: images})


  predictor_fn = tf.contrib.predictor.from_saved_model(
      FLAGS.model_dir,
      signature_def_key=FLAGS.signature_def_key,
      tags=FLAGS.tags)
  times = []
  for _ in xrange(FLAGS.num_runs):
    start = time.time()
    predictor_fn({FLAGS.input_name: batch})
    end = time.time()
    times.append(end - start)
  if FLAGS.output_tsv is not None:
    with open(FLAGS.output_tsv, 'w') as fout:
      writer = csv.writer(fout, delimiter='\t')
      writer.writerows([[t] for t in times])
  print('Min time: %s' % min(times))
  print('Median time: %s' % np.median(times))

def main(unused_argv):

  return 0

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
gpu_options.allow_growth = True

def load_model(model_dir):
  tf.reset_default_graph()
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
  logging.info('Loading model ...')
  metagraph = tf.saved_model.loader.load(
      sess, [tf.saved_model.tag_constants.SERVING], model_dir)
  logging.info('Getting signature ...')
  signature = metagraph.signature_def[
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  logging.info(signature)
  outputs = [v.name for v in signature.outputs.values()]
  logging.info(outputs)
  indexes = {k: i for i, k in enumerate(signature.outputs.keys())}
  logging.info(indexes)
  return sess, signature


def run_inference(sess, signature, image_files, output_file):
  outputs = [v.name for v in signature.outputs.values()]
  indexes = {k: i for i, k in enumerate(signature.outputs.keys())}
  with open(output_file, 'w') as fp:
    for ii, image_filename in enumerate(image_files):
      with open(image_filename) as ff:
        image_bytes = ff.read()
      res = sess.run(
        fetches=outputs,
        feed_dict={
          signature.inputs['image_bytes'].name: [image_bytes],
          signature.inputs['key'].name: [str(ii)]
        })
      scores =  res[indexes['scores']][0]
      #labels = res[indexes['labels']][0]
      predictions = []
      for i in range(len(scores)):
        #predictions.append("%s: %.3f"% (labels[i], scores[i]))
        predictions.append("%.3f"% (scores[i]))

      ll = '%s, %s' % (image_filename, ', '.join(predictions))
      print ll
      fp.write(ll + '\n')


def run_batch_inference(sess, signature, image_files, output_file):
  outputs = [v.name for v in signature.outputs.values()]
  indexes = {k: i for i, k in enumerate(signature.outputs.keys())}

  image_bytes_tensor = []
  image_key_tensor = []
  for ii, image_filename in enumerate(image_files):
    with open(image_filename) as ff:
      image_bytes_tensor.append(ff.read())
      image_key_tensor.append(str(ii))

  res = sess.run(
    fetches=outputs,
    feed_dict={
      signature.inputs['image_bytes'].name: image_bytes_tensor,
      signature.inputs['key'].name: image_key_tensor
    })

  with open(output_file, 'w') as fp:
    for i in range(len(res[indexes['scores']])):
      scores =  res[indexes['scores']][i]
      #labels = res[indexes['labels']][i]
      predictions = []
      for j in range(len(scores)):
        #predictions.append("%s: %.3f"% (labels[j], scores[j]))
        predictions.append("%.3f"% (scores[j]))
      ll = '%s, %s' % (image_files[i], ', '.join(predictions))
      print ll
      fp.write(ll + '\n')
