from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

import tensorflow as tf
import numpy as np

def func1():
  a = tf.constant(3.0, dtype=tf.float32)
  b = tf.constant(4.0, dtype=tf.float32)
  total = a+b
  print(a)
  print(b)
  print(total)
  writer = tf.summary.FileWriter('.')
  writer.add_graph(tf.get_default_graph())
  sess = tf.Session()
  print(tf.Session().run({'ab':(a,b)}))

def func2():
  ### check how to specify shape
  vec = tf.random_uniform(shape=(3,))
  s = tf.shape(vec)
  print(tf.Session().run(s))
  print(vec)
  print(vec.shape)
  out1 = vec + 1
  out2 = vec + 2
  print(tf.Session().run(vec))
  print(tf.Session().run(vec))
  print(tf.Session().run((out1, out2)))

def func3():
  x = tf.placeholder(tf.float32)
  y = tf.placeholder(tf.float32)
  z = x + y
  print(tf.Session().run(z, feed_dict={x:3, y:4}))
  print(tf.Session().run(z, feed_dict={x:[1, 3], y:[2, 5]}))

def func4():
  test_data = [
    [0, 1],
    [2, 3],
  ]
  slices = tf.data.Dataset.from_tensor_slices(test_data)
  next_item = slices.make_one_shot_iterator().get_next()
  session = tf.Session()
  while True:
    try:
      print(session.run(next_item))
    except tf.errors.OutOfRangeError:
      break;

def func5():
  r = tf.random_normal([10, 3])
  dataset = tf.data.Dataset.from_tensor_slices(r)
  iterator = dataset.make_initializable_iterator()
  next_item = iterator.get_next()

  ## Error: Cannot capture a stateful node, normal function is stateful
  #next_item = dataset.make_one_shot_iterator().get_next()
  session = tf.Session()
  session.run(iterator.initializer)
  while True:
    try:
      print(session.run(next_item))
    except tf.errors.OutOfRangeError:
      break;

def func6():
  x = tf.placeholder(tf.float32, shape=[None, 3])
  linear_model = tf.layers.Dense(units=1,
      kernel_initializer=tf.zeros_initializer())
  y = linear_model(x)
  init = tf.global_variables_initializer()
  session = tf.Session()
  session.run(init)
  print(session.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


def func7():
  features = {
    'sales':[[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']
  }
  department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening'])
  department_column = tf.feature_column.indicator_column(department_column)
  columns = [tf.feature_column.numeric_column('sales'),
             department_column]
  inputs = tf.feature_column.input_layer(features, columns)
  var_init = tf.global_variables_initializer()
  table_init = tf.tables_initializer()
  session = tf.Session()
  session.run((var_init, table_init))
  print(session.run(inputs))

def func8():
  ## shape: (4,1)
  x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
  ## shape: (4,1)
  y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
  print('=======shape of ground truth =========')
  print(y_true)
  print('=======shape of ground truth =========')
  ## shape: (4,1), if we change units = 1 to units = 2 then the shape will
  ## be (4, 2)
  linear_model = tf.layers.Dense(units=1,
                                 kernel_initializer=tf.zeros_initializer())
  y_pred = linear_model(x)
  print('=======shape of prediction =========')
  print(y_pred)
  print('=======shape of prediction =========')
  weights = linear_model.get_weights()
  loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train_op = optimizer.minimize(loss)
  y_pred_post = linear_model(x)
  loss_post = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

  session = tf.Session()
  init = tf.global_variables_initializer()
  session.run(init)

  for i in range(2):
    print ('======= iteration {} ======'.format(i))
    print(session.run(loss))
    #### loss is evaluated before train_op
    _, loss_value = session.run((train_op, loss))
    #for v in tf.trainable_variables():
    for v in linear_model.trainable_variables:
      print(v.name)
      print(session.run(v[0]))
    print(loss_value)

  print(session.run(y_pred))
  print(session.run(loss))

def func9():
   x = tf.constant([[92, 4], [7, 6]], dtype=tf.float32)
   y = tf.argmax(x, axis=0)
   session = tf.Session()
   r = tf.rank(x)
   print(session.run(x))
   print(session.run(y))
   print(session.run(r))


def func10():
  y = tf.Variable([6])
  y = y.assign([7])
  with tf.Session() as sess:
     print(sess.run(y))

def func11():
  y_true = tf.constant([[0, 0, 0]], dtype=tf.int32)
  logits = tf.constant([[0.9, 0, 0]], dtype=tf.float32)

  loss = tf.losses.softmax_cross_entropy(y_true, logits)

  session = tf.Session()
  init = tf.global_variables_initializer()
  session.run(init)
  print(session.run(loss))

def func12():
  labels= tf.constant([-1, -1, -1], dtype=tf.int32)
  less = tf.greater_equal(labels, tf.zeros_like(labels))
  mask = tf.cast(tf.greater_equal(labels, tf.zeros_like(labels)), labels.dtype)
  weights = tf.multiply(
      tf.ones_like(labels),
      tf.cast(tf.greater_equal(labels, tf.zeros_like(labels)), labels.dtype))

  session = tf.Session()
  init = tf.global_variables_initializer()
  session.run(init)
  print(session.run(weights))

def test_conv2d():
  features = tf.constant(np.array([[1, 2], [2, 5], [3, 6], [4, 7]]), dtype=tf.float32)
  weights = np.array([[0.2, 0.5], [0.3, 0.7]])
  features = np.array([[1, 2], [2, 5], [3, 6], [4, 7]])
  logits = tf.layers.dense(tf.constant(features, dtype=tf.float32),
                           units=2,
                           kernel_initializer=tf.constant_initializer(weights),
                           bias_initializer=tf.zeros_initializer)
  expected_logits = features.dot(weights)
  print(expected_logits)

  ## simulate feature length from inception v3.
  features2 = np.array([[[[1, 2]]], [[[2, 5]]], [[[3, 6]]], [[[4, 7]]]])
  conv = tf.contrib.layers.conv2d(tf.constant(features2, dtype=tf.float32),
                   num_outputs=2,
                   kernel_size=[1,1],
                   weights_initializer=tf.constant_initializer(weights, dtype=tf.float32),
                   biases_initializer=tf.zeros_initializer,
                   activation_fn=None,
                   normalizer_fn=None)
  conv = tf.squeeze(conv, [1, 2], name='SpatialSqueeze')

  print(conv)
  init = tf.global_variables_initializer()
  with tf.Session() as session:
    session.run(init)
    print(session.run(logits))
    print(session.run(conv))

def test_example_proto():
  image = 'aa'
  example = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'image/encoded':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[image]))
            }))
  print('aaa')
  print(example)
  example.SerializeToString()

def test_tensor_indexing():
  labels= tf.math.exp((tf.Variable([[-1, -1, -1], [1, 0, 1], [2, 5, 7]], dtype=tf.float32)))
  x = tf.constant([[5, 6, 8], [2, 3, 4], [4, 6, 9]], dtype=tf.float32)
  max_ = tf.maximum(x[0:, 1:2], x[0:,2:])
  min_ = tf.minimum(max_, labels[0:, 1:2])
  labels = labels[0:, 1:2].assign(min_)


  session = tf.Session()
  init = tf.global_variables_initializer()
  session.run(init)
  print(session.run(labels))


def test_tensor_update():
  zeros = tf.zeros([10, 300], tf.float32)
  # Update the 0-th row of features with all ones
  update_op = tf.tensor_scatter_update(zeros, 0, [1.0]*300)
  init = tf.global_variables_initializer()
  with tf.Session() as session:
    session.run(init)
    session.run(update_op)
    print(zeros.eval())


def test_per_row_update():
  ref_tensor = tf.constant([[2], [5], [1]], dtype=tf.float32)
  input_tensor = tf.constant([[5, 6, 8], [2, 3, 4], [4, 6, 9]],
                              dtype=tf.float32)


  columns_to_be_updated = []
  for i in range(input_tensor.get_shape()[1]):
    current_column = input_tensor[0:, i:i + 1]
    min_prob = tf.transpose(tf.minimum(ref_tensor, current_column))
    columns_to_be_updated.append(min_prob)
  output_tensor = tf.transpose(tf.concat(columns_to_be_updated, 0))
  with tf.Session() as session:
    print(session.run(output_tensor))
  """
  print('=================')
  print(columns_to_be_updated)
  output_tensor = tf.concat(columns_to_be_updated, 0)
  print(output_tensor)
  print('=================')
  with tf.Session():
    print(output_tensor)
    self.assertTrue(
        np.allclose(output_tensor.eval(),
                    np.array([[5, 6, 8], [2, 3, 4], [4, 6, 9]])))

  """



def test_string_tensor():
  input_feature = {'input':tf.constant("test1", dtype=tf.string)}
  def true_fn():
    return tf.constant(1, dtype=tf.int32)
  def false_fn():
    return tf.constant(0, dtype=tf.int32)
  output = tf.cond(tf.equal(input_feature['input'], "test"), lambda: tf.constant(1, dtype=tf.int32), lambda:tf.constant(0, dtype=tf.int32))
  #output_feature = input_feature

  with tf.Session() as session:
    print(session.run(output))

def main(unused_argv):
  test_string_tensor()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()


