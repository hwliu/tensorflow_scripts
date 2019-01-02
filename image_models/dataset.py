import tensorflow as tf
import json

class Dataset(object):
  def __init__(self, json_data):
    self.dataset_name = json_data['dataset_name']
    self.num_classes = json_data['num_classes']
    self.num_channels = json_data['num_channels']
    self.is_multilabel = json_data['is_multilabel']
    self.training_dataset_path = json_data['dataset_split']['train']['filenames']
    self.training_dataset_total_images = json_data['dataset_split']['train']['num_images']
    self.validation_dataset_path = json_data['dataset_split']['test']['filenames']
    self.validation_dataset_total_images = json_data['dataset_split']['test']['num_images']


def create_dataset_from_json_file(path_to_json_file):
  with open(path_to_json_file) as f:
    dataset_metadata = json.load(f)
  return Dataset(dataset_metadata)


