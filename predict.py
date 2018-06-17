"""predict data"""

import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np
from model.utils import text2vec


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")

some_dict = {'eval_labels': ("that sounds awesome . i'm just washing my cars and my dogs on my day off .",),
             'label_candidates': ('i read anything with spiderman !',
                                  'i like jazz but my best friend and now roommate plays rock all the time',
                                  "yea that's the problem with blueberries lol",
                                  "i'm okay , eating some ice cream before bed .",
                                  'pizza with black olives .',
                                  "i'm very impressed ! you seem to have it all together .",
                                  'i will need a bedroom painted in purple and lime green though .',
                                  'i watch kids on the playground . i teach third grade .',
                                  'i play video games . would love to make a living out of it',
                                  'dang that kinda weird . well do you have a pet',
                                  'well , anyone can sing , lessons help',
                                  "she's 9 . i love them all . what work did you do before retiring ?",
                                  "hello , i'm doing very well , thank you . i'm getting married soon .",
                                  'oh i want to visit there very badly , but my cancer just went into remission',
                                  'you are married ? i will finish high school in two years .',
                                  "do you have pet ? i have 5 children and that's more then enough . all boys",
                                  'hello ! you should friend me on facebook .',
                                  'what games do you play ?',
                                  'chemistry was my favorite subject , did you get bullied at all ?',
                                  "that sounds awesome . i'm just washing my cars and my dogs on my day off ."),
             'text': "your persona: i like to go to country concerts on weekends.\nyour persona: i own two vintage mustangs.\nyour persona: my favorite music is country.\nyour persona: i've two dogs.\nhello there ! i'm reginald , or reggie if you prefer . how are you today ?\nhowdy , hey ! i'm pretty good\nthat's great . i'm doing good as well . the grandkids are coming over later"}

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # ПОДГРУЖАЕМ ПАРАМЕТРЫ
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # ОПРЕДЕЛЯЕМ МОДЕЛЬ
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.pkl'), 'rb'))

    test_data, true_id, true_ans, raw_dial, cands = text2vec(some_dict, word2idx)

    # подаем по 20 кандидатов и находим лучшие из них
    test_input_fn = tf.estimator.inputs.numpy_input_fn(test_data,
                                                       num_epochs=1,
                                                       batch_size=20,
                                                       shuffle=False)

    test_predictions = estimator.predict(test_input_fn,
                                         predict_keys=['y_prob'],
                                         yield_single_examples=False)

    for i, batch in enumerate(test_predictions):
        best_id = 10 * i + np.argmax(batch['y_prob'])
        sorted_elements = np.argsort(batch['y_prob'])[::-1]
        print('Most likely answer id = {}'.format(best_id))
        print('True id: {}'.format(true_id))
        print('Sorted elements: {}'.format(sorted_elements))
        print('---------------------------------------------------')
        print('Dialog: ')
        for mes in raw_dial:
            print(mes)
        print('---------------------------------------------------')
        print('True answer: {}'.format(true_ans))
        print('Predicted answer: {}'.format(cands[best_id]))
        # print('Best raw reply: {}'.format(raw_test_data[best_id]))
