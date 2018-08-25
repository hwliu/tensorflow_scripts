"""Define the model for inception v3 model."""

import tensorflow as tf

class LoggerHook(tf.train.SessionRunHook):
   "Logs model training"

   def before_run(self, run_context):
     print ('before run.....')
     variables = tf.trainable_variables()
     print (variables)
     return tf.train.SessionRunArgs(variables)

   def after_run(self, run_context, run_values):
     print ('after run.....')
     print run_values
     return 0

   def set_variable_name(self, value):
     self._variables_to_log = value

def CreateLogger(variable_names_to_tensors):
   h = LoggerHook()
   h.set_variable_name(variable_names_to_tensors)
   print(variable_names_to_tensors)
   return h


