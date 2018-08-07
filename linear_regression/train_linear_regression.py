import tensorflow as tf
import tensorflow.feature_column as fc
import pandas

import functools
import os
import sys

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

def load_dataset(dataset_dir):
  train_file = os.path.join(dataset_dir, 'adult.data')
  test_file = os.path.join(dataset_dir, 'adult.test')
  train_df = pandas.read_csv(train_file, header = None, names = _CSV_COLUMNS)
  test_df = pandas.read_csv(test_file, header = None, names = _CSV_COLUMNS)
  return [train_df, test_df, train_file, test_file]

def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
  label = df[label_key]
  ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
  if shuffle:
    ds = ds.shuffle(10000)
  ds = ds.batch(batch_size).repeat(num_epochs)
  return ds

def better_input_fn(data_file, num_epochs, shuffle, batch_size):
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run census_dataset.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    classes = tf.equal(labels, '>50K')  # binary classification
    return features, classes

  dataset = dataset.map(parse_csv, num_parallel_calls=5)
  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

def build_feature_columns():
  age = fc.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')

  relationship = fc.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

  occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)

  education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

  age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  education_x_occupation = tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)

  age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)

  base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
  ]

  return [base_columns, crossed_columns]


def main(unused_argvs):
  DATASET_DIR = './dataset/'
  [train_df, test_df, train_file, test_file] = load_dataset(DATASET_DIR)
  print 'printing training data with naive method..'
  print(train_df)

  #ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)
  #ds = better_input_fn(train_file, num_epochs=5, shuffle=True, batch_size=10)
  #for feature_batch, label_batch in ds.take(1):
  #  print('Some feature keys:', list(feature_batch.keys())[:5])
  #  print()
  #  print('A batch of Ages  :', feature_batch['age'])
  #  print()
  #  print('A batch of Labels:', label_batch )
  train_inpf = functools.partial(better_input_fn, train_file, num_epochs=40, shuffle=True, batch_size=64)
  test_inpf = functools.partial(better_input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

  [base_columns, crossed_columns] = build_feature_columns()

  classifier = tf.estimator.LinearClassifier(feature_columns=base_columns + crossed_columns,
                                            optimizer=tf.train.FtrlOptimizer(learning_rate=0.1))
  classifier.train(train_inpf)
  result = classifier.evaluate(test_inpf)
  for key,value in sorted(result.items()):
    print('%s: %s' % (key, value))

if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run(main)

