"""Define the model for inception v3 model."""

import tensorflow as tf

class LoggerHook(tf.train.SessionRunHook):
   "Logs model training"

   def before_run(self, run_context):
     print('before run')
     print(tf.trainable_variables())
     for v in tf.trainable_variables():
       if v.name == 'InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/weights:0':
         print('found variable')
         variables=v

     #variables = self._variables_to_log
     print (variables)
     return tf.train.SessionRunArgs(variables)

   def after_run(self, run_context, run_values):
     print ('after run.....')
     print(run_values)
     return 0

   def set_variable_name(self, value):
     self._variables_to_log = value

def CreateLogger(variable_list=None):
   h = LoggerHook()
   h.set_variable_name(variable_list)
   return h


