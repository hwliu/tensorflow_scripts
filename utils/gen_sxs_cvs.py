from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

from os import walk

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_folder',
    '/media/haoweiliu/Data/cropping/sxs/exp',
    'The score file to plot precision recall curve.')

flags.DEFINE_string(
    'output_csv_file', '/media/haoweiliu/Data/cropping/sxs/images.csv',
    'The path hold the precision recall curve plot.')

def main(unused_argv):
  f = []
  for (dirpath, dirnames, filenames) in walk(FLAGS.input_folder):
    f.extend(filenames)
    break

  csvf = open(FLAGS.output_csv_file, "w")
  str_to_write = 'a.description, i.urlB, i.imageB, i.urlE, i.imageE\n'
  csvf.write(str_to_write)

  for filename in f:
    url_baseline = 'https://www.google.com/evaluation/result/static/e/baseline/baseline/{}'.format(filename)
    url_exp = 'https://www.google.com/evaluation/result/static/e/exp/exp/{}'.format(filename)
    str_to_write = '{}, {}, {}, {}, {}\n'.format(filename, url_baseline, url_baseline, url_exp, url_exp)
    csvf.write(str_to_write)
  csvf.close()

  return 0

if __name__ == '__main__':
  app.run(main)
