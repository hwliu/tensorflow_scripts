from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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
   ## shape: (4,1)
  x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
  ## shape: (4,1)
  y_true = tf.constant([[0], [1], [0], [1]], dtype=tf.int32)
  print('=======shape of ground truth =========')
  print(y_true)
  print('=======shape of ground truth =========')
  ## shape: (4,1), if we change units = 1 to units = 2 then the shape will
  ## be (4, 2)
  logits = tf.layers.dense(x, units=2)
  print(logits)
  loss = tf.losses.sparse_softmax_cross_entropy(y_true, logits)

  session = tf.Session()
  init = tf.global_variables_initializer()
  session.run(init)
  print(session.run(loss))
  print(session.run(logits))

def main(unused_argv):
  func11()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()


