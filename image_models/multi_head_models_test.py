import tensorflow as tf
from multi_head_models import get_model_fn

class UtilTest(tf.test.TestCase):
   def test_add_final_layer(self):
     model_fn = get_model_fn()
     model_fn(None, None, None)
     self.assertTrue(True)

if __name__ == '__main__':
  tf.test.main()
