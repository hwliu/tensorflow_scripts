"""Define the model for inception v3 model."""

import tensorflow as tf

class LoggerHook(tf.train.SessionRunHook):
   "Logs model training"

   def before_run(self, run_context):
     print('before run.......')
     print (self._variables_to_log)
     return tf.train.SessionRunArgs(self._variables_to_log)

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


