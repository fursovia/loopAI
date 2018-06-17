import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np
import copy
from model.utils import text2vec
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from tensorflow.python.client import device_lib


class serving_input_fn:
    def __init__(self):
        self.features = tf.placeholder(tf.int64, shape=[None, 200])
        self.receiver_tensors = {
           'text': self.features,
        }
        self.receiver_tensors_alternatives = None


class DSSMAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)

        agent = argparser.add_argument_group('DSSM Arguments')
        agent.add_argument('--model_dir', default='experiments',
                    help="Experiment directory containing params.json")
        agent.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")

    def __init__(self, opt):
        super().__init__(opt)
        self.id = 'DSSM'
        self.observation = {}
        self.episode_done = True
        self.estimator = self.create_model()
        self.opt = opt
        self.create_predictor()

    def txt2vec(self, txt):
        return np.array(self.dict.txt2vec(txt)).astype(np.int32)

    def vec2txt(self, vec):
        return self.dict.vec2txt(vec)

    def observe1(self, observation):
        observation = copy.deepcopy(observation)

        if not self.episode_done:
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']

        self.observation = observation
        self.episode_done = observation['episode_done']

        return observation

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        batch_idx = self.opt.get('batchindex', 0)
        self.observation = obs
        #self.answers[batch_idx] = None
        return obs


    def create_model(self):

        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        # ПОДГРУЖАЕМ ПАРАМЕТРЫ
        json_path = os.path.join(self.opt['model_dir'], 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        self.params = Params(json_path)

        # ОПРЕДЕЛЯЕМ МОДЕЛЬ
        tf.logging.info("Creating the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230, model_dir=self.opt['model_dir'])

        estimator = tf.estimator.Estimator(model_fn, params=self.params, config=config)

        return estimator


    def create_predictor(self):
        self.predictor = tf.contrib.predictor.from_estimator(
            self.estimator,
            serving_input_fn
        )


    def predict(self, some_dicts):
        word2idx = pickle.load(open(os.path.join(self.opt['data_dir'], 'word2idx.pkl'), 'rb'))

        data_to_predict = []
        candidates = []
        lengis = []

        for dict_ in some_dicts:
            test_data, true_id, true_ans, raw_dial, cands = text2vec(dict_, word2idx)
            data_to_predict.append(test_data)
            lengis.append(len(test_data))
            candidates.append(cands)

        data_to_predict = np.array(data_to_predict, int).reshape(-1, 200)

        test_predictions = self.predictor({'text': data_to_predict})['y_prob']

        output = []
        for i, leng in enumerate(lengis):
            sorted_elements = np.argsort(test_predictions[:leng])[::-1]
            cands = np.array(candidates[i], object)
            ppp = cands[sorted_elements]
            output.append(ppp)
            test_predictions = test_predictions[leng:]

        return output

    def batch_act(self, observations):
        # observations:
        #       [{'label_candidates': {'office', ...},
        #       'episode_done': False, 'text': 'Daniel ... \nWhere is Mary?',
        #       'labels': ('office',), 'id': 'babi:Task10k:1'}, ...]

        batchsize = len(observations)
        batch_reply = [{'id': self.id} for _ in range(batchsize)]

        predictions = self.predict(observations)

        for i in range(len(batch_reply)):
            batch_reply[i]['text_candidates'] = predictions[i]
            batch_reply[i]['text'] = batch_reply[i]['text_candidates'][0]

        return batch_reply # [{'text': 'bedroom', 'id': 'RNN'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]
