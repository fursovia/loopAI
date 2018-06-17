"""data iterator"""

import tensorflow as tf
import os
from model.utils import decode


def train_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    dataset = tf.data.TFRecordDataset(os.path.join(data_dir, 'train.tfrecords'))
    dataset = dataset.shuffle(params.train_size)
    dataset = dataset.map(decode)

    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None)

    return dataset

def eval_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    dataset = tf.data.TFRecordDataset(os.path.join(data_dir, 'eval.tfrecords'))
    dataset = dataset.map(decode)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None)

    return dataset


def final_train_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = tf.data.TFRecordDataset(os.path.join(data_dir, 'full.tfrecords'))
    dataset = dataset.shuffle(params.train_size)
    dataset = dataset.map(decode)

    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None)

    return dataset
