"""Обучаем модель"""

import argparse
import os

import tensorflow as tf
from model.input_fn import train_input_fn
from model.input_fn import eval_input_fn
from model.input_fn import final_train_input_fn
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/sample',
                    help="Directory containing the dataset")
parser.add_argument('--final_train', default='N',
                    help="Whether to train on a whole dataset")
parser.add_argument('--num_gpus', type=int, default=1,
                    help="Number of GPUs to train on")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    if args.num_gpus > 1:
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)
    else:
        distribution = None

    session_config = tf.ConfigProto(device_count={'GPU': 1})
    session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    config = tf.estimator.RunConfig(tf_random_seed=43,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps,
                                    train_distribute=distribution,
                                    session_config=session_config)

    estimator = tf.estimator.Estimator(model_fn,
                                       params=params,
                                       config=config)

    if args.final_train == 'N':
        # Train the model
        tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
        estimator.train(lambda: train_input_fn(args.data_dir, params))

        # Evaluate the model on the valid set
        tf.logging.info("Evaluation on valid set.")
        res = estimator.evaluate(lambda: eval_input_fn(args.data_dir, params))

        for key in res:
            print("{}: {}".format(key, res[key]))
    else:
        # Train the model
        tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
        estimator.train(lambda: final_train_input_fn(args.data_dir, params))

        # Evaluate the model on the valid set
        tf.logging.info("Evaluation on valid set.")
        res = estimator.evaluate(lambda: eval_input_fn(args.data_dir, params))

        for key in res:
            print("{}: {}".format(key, res[key]))
