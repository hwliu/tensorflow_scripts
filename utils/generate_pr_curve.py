"""Script to generate a precision-recall plot.

The script generates a precision-recall plot given a scoring file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_class',
    4,
    'The score file to plot precision recall curve.')

flags.DEFINE_string(
    'score_file',
    '/media/haoweiliu/Data/image_attribute_score_files/bgtype_score.txt',
    'The score file to plot precision recall curve.')

flags.DEFINE_string(
    'output_pr_plot_path', '/media/haoweiliu/Data/pr_curves',
    'The path hold the precision recall curve plot.')

flags.DEFINE_string(
    'output_proto_path', '/media/haoweiliu/Data/protos',
    'The path hold the precision recall curve plot.')




def get_labels_and_scores_all(filename):
  """Load a text file and returns image name, scores, and labels."""
  [_, _, positive_scores, predicted_label, ground_truth_label] = numpy.loadtxt(
      filename,
      unpack=True,
      dtype={
          'names': ('filename', 'negative score', 'positive score',
                    'predicted_label', 'ground_truth_label'),
          'formats': ('S200', 'f4', 'f4', 'i4', 'i4')
      })
  return (predicted_label, ground_truth_label, positive_scores)

def get_all_scores_and_labels(filename, num_class):
  """Load a text file and returns image name, scores, and labels."""
  class_names = tuple('class{}_score'.format(n) for n in range(num_class))
  class_formats = tuple('f4' for n in range(num_class))
  field_names = ('filename', ) + class_names + ( 'predicted_label', 'ground_truth_label')
  format_names = ('S200',) + class_formats + ('i4', 'i4')

  alldata = numpy.loadtxt(
      filename,
      unpack=True,
      dtype={
          'names': field_names,
          'formats': format_names
      })
  num_fields = len(alldata)
  class_scores = alldata[1:1+num_class]
  ground_truth_label = alldata[len(alldata)-1]
  predicted_label = alldata[len(alldata)-2]
  return (class_scores, predicted_label, ground_truth_label)

def compute_pr_curve(labels, positive_scores):
  return precision_recall_curve(labels, positive_scores)

def save_pr_plot(precision, recall, app_precision, app_recall, pr_plot_label, output_pr_plot_path):
  fig = plt.figure()
  ax1 = fig.add_axes((0.1, 0.3, 0.8, 0.6))

  ax1.plot(
      recall[:-2], precision[:-2], color='blue', label='true_pr')
  ax1.plot(
      app_recall[:-2], app_precision[:-2], color='red', label='approximated pr')
  ax1.set_ylabel('Precision')
  ax1.set_xlabel('Recall')
  ax1.set_xlim(0.1, 1)
  ax1.legend(loc='lower left')
  ax1.set_title(
      'Precision-Recall curves.', fontsize=10)

  plt.savefig(
      output_pr_plot_path,
      format='png',
      bbox_inches='tight',
      pad_inches=0.02,
      dpi=150)

def gen_approximate_class_score(class_scores, target_class):
  num_classes = class_scores.shape[0]
  new_scores = class_scores[target_class, :].copy()
  max_classes = numpy.argmax(class_scores, axis=0)
  max_scores = numpy.max(class_scores, axis=0)
  total_scores = numpy.sum(class_scores, axis=0)
  indices = numpy.where(max_classes!=target_class)
  val = 1 - (num_classes-1) * max_scores[indices]
  print('*******************')
  print(len(numpy.where(max_classes==target_class)[0]))
  print(len(indices[0]))
  print(len(numpy.where(val>=0)[0]))
  print(len(total_scores))
  print(total_scores)
  print('*******************')
  new_scores[indices] = total_scores[indices] - (num_classes-1) * max_scores[indices]
  new_scores[numpy.where(new_scores < 0)] = 0
  return new_scores

def write_proto_file(precision, recall, threshold, step, proto_file):
  #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
  num_point = len(threshold)
  threshold = threshold[0::step]
  precision = precision[0:num_point:step]
  recall = recall[0:num_point:step]
  print(len(precision))
  print(len(recall))
  print(len(threshold))

  f = open(proto_file, "w")
  pre_confidence = -1
  pre_r = -1
  for p, r, t in zip(reversed(precision), reversed(recall), reversed(threshold)):
    confidence = int(t*16383)
    if confidence != pre_confidence and r != pre_r:
       str_to_write = 'points {{ precision: {} recall: {} min_confidence: {} }}\n'.format(p, r, confidence)
       pre_confidence=confidence
       pre_r = r
       f.write(str_to_write)
  f.close()
  return 0

def main(unused_argv):
  num_classes = FLAGS.num_class
  output_pr_plot_path = FLAGS.output_pr_plot_path
  output_proto_path = FLAGS.output_proto_path
  [class_scores, predicted_label, ground_truth_label] = get_all_scores_and_labels(FLAGS.score_file, num_classes)
  class_scores = numpy.vstack(class_scores)

  for n in range(num_classes):
    label = (ground_truth_label==n)*1
    scores = class_scores[n, :]
    [precision, recall, threshold] = compute_pr_curve(label, scores)
    area = auc(recall, precision)
    approximated_scores = gen_approximate_class_score(class_scores, n)
    [app_precision, app_recall, app_threshold] = compute_pr_curve(label, approximated_scores)
    print('=========================================')
    print(len(app_precision))
    print(len(app_recall))
    print(len(app_threshold))
    print('Area Under PR Curve(AP): %0.2f' % area)
    print('Generating PR curve for class {}'.format(n))
    pr_plot_label = 'class{}'.format(n)
    pr_plot_file = output_pr_plot_path + '/' + pr_plot_label + '.png'
    save_pr_plot(precision, recall, app_precision, app_recall, pr_plot_label, pr_plot_file)
    print('=========================================')
    proto_file = output_proto_path + '/' + pr_plot_label + '.textproto'
    write_proto_file(app_precision, app_recall, app_threshold, 1, proto_file)

if __name__ == '__main__':
  app.run(main)
