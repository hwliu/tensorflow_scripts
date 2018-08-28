"""Define the model for inception v3 model."""

import tensorflow as tf
import json

class LoggerHook(tf.train.SessionRunHook):
   "Logs model training"

   def before_run(self, run_context):
     print('before run.......')
     print(self._variables_to_log)
     #dict_to_log = {}
     #for v in self._variables_to_log:
     #  dict_to_log[v.name] = v
     return tf.train.SessionRunArgs(self._variables_to_log)

   def after_run(self, run_context, run_values):
     #f = open('/media/haoweiliu/Data/scratch_models/model_parametes.txt', 'w')
     #results = run_values.results
     #for key in sorted(results):
     #    f.write('============================================================\n')
     #    f.write(str(key))
     #    f.write('\n')
     #    f.write(str(results[key]))
     #    f.write('\n')
     #    f.write('============================================================\n')
     #f.close()
     print(run_values)
     return 0

   def set_variable_name(self, value):
     self._variables_to_log = value

def CreateLogger(variable_list=None):
   h = LoggerHook()
   h.set_variable_name(variable_list)
   return h


