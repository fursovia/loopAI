"""Вспомогательные функции"""

import json
import os
import numpy as np
import pickle
import tensorflow as tf
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import ngrams
from keras.preprocessing.sequence import pad_sequences


snowball_stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.num_epochs = None
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def get_coefs(arr):
    word, arr = arr[:-300], arr[-300:]
    return ' '.join(word), np.array(arr, dtype=np.float64)


def get_embeddings(params):
    word2idx_file = os.path.join(params.data_path, 'word2idx.pkl')
    fasttext_file = os.path.join(params.data_path, 'fasttext.vec')

    word2idx = pickle.load(open(word2idx_file, 'rb'))
    embeddings_index = dict(get_coefs(o.strip().split()) for o in open(fasttext_file, encoding='utf-8'))

    embedding_matrix = np.zeros(((params.vocab_size + 1), params.embedding_size))

    for word, i in word2idx.items():
        if type(word) == tuple:
            word = ' '.join(word)

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.1, 0.1, params.embedding_size)

    return embedding_matrix


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_records(data, name, save_to):
    """Converts a dataset to tfrecords."""
    X, Y = data
    num_examples = Y.shape[0]

    filename = os.path.join(save_to, name + '.tfrecords')
    print('Writing...', filename)

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            Y_ = Y[index]
            X_ = X[index]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature([Y_]),
                        'sent': _int64_feature(X_)
                    }
                )
            )
            writer.write(example.SerializeToString())


def decode(serialized_example):
    """Parses an image and label from the given serialized_example."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'sent': tf.FixedLenFeature([200], tf.int64)
        })

    label = tf.cast(features['label'], tf.int64)
    sent = tf.cast(features['sent'], tf.int64)

    return sent, label


def clean(text, with_stopwords=True):
    text = text.strip().lower()
    text = re.sub('[^\w\s]', ' ', text)
    text = re.sub('^\d+\s|\s\d+\s|\s\d+$', " <num> ", text)
    if with_stopwords:
        return ' '.join(snowball_stemmer.stem(word) for word in text.split())
    else:
        return ' '.join(snowball_stemmer.stem(word) for word in text.split() if word not in stop_words)


def vectorize_text(text, word2idx, maxlen=20, truncating_type='post', only_words=True):
    vec_seen = []

    words_ = text.split()

    for word in words_:
        try:
            vec_seen.append(word2idx[word])
        except:
            continue

    if not only_words:
        bi_ = ngrams(words_, 2)
        for bi in bi_:
            try:
                vec_seen.append(word2idx[bi])
            except:
                continue

    return pad_sequences([vec_seen], maxlen=maxlen, truncating=truncating_type)[0]


def text2vec(dict_from_parlai, word2idx):
    personal_info = []  # нужно иметь 5 фактов
    dial = []
    raw_dial = []
    cands = dict_from_parlai['label_candidates']
    splitted_text = dict_from_parlai['text'].split('\n')
    true_ans = dict_from_parlai['eval_labels'][0]
    for i, cand in enumerate(cands):
        if cand == true_ans:
            true_answer_id = i

    cleaned_cands = []
    for mes in cands:
        cleaned_cands.append(clean(mes))

    for mes in splitted_text:
        if 'your persona:' in mes:
            personal_info.append(clean(' '.join(mes.split(':')[1:])))
        else:
            dial.append(clean(mes))
            raw_dial.append(mes)

    dial_len = len(dial)

    if dial_len == 1:
        cont = ''
        quest = dial[0]

    if dial_len == 2:
        cont = dial[0]
        quest = dial[1]

    if dial_len == 3:
        cont = ' '.join(dial[0:2])
        quest = dial[2]

    if dial_len > 3:
        cont = ' '.join(dial[dial_len-4:dial_len-1])  # контекст длины 3
        quest = dial[dial_len-1]

    info_5 = []
    for i in range(5):
        try:
            info_5.append(personal_info[i])
        except IndexError:
            info_5.append('')

    X = []
    for cand in cleaned_cands:
        X.append([cont, quest, cand, info_5])

    context_vect = []
    question_vect = []
    reply_vect = []
    info_vect = []
    for i, dial in enumerate(X):
        cont = vectorize_text(dial[0], word2idx, 60, 'pre')
        ques = vectorize_text(dial[1], word2idx)
        reply = vectorize_text(dial[2], word2idx)

        context_vect.append(cont)
        question_vect.append(ques)
        reply_vect.append(reply)

        info_ = dial[3]
        for j in range(5):  # 5 фактов о каждом
            vect_info = vectorize_text(info_[j], word2idx)
            info_vect.append(vect_info)

    data = np.hstack((context_vect, question_vect, reply_vect, np.array(info_vect).reshape(-1, 100)))

    return data, true_answer_id, cands[true_answer_id], raw_dial, cands